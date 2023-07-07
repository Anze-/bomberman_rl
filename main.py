import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

import neat
import pickle

GENERATION = 0

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)


def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval,
                     skip_end_round=False):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]


            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                # QUI ESEGUE GLI STEP!!
                world.do_step(user_input, gui)
                user_input = None
            else:
                # Might want to wait
                pass

        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if skip_end_round:
            if gui is not None:
                do_continue = False
                while not do_continue:
                    render(True)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        elif event.type == pygame.KEYDOWN:
                            key_pressed = event.key
                            if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                                do_continue = True

    world.end()


def main(argv=None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS,
                             help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--train_genetic", default=False, action="store_true",
                             help="Trains a genetic agent to play bomberman")

    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int,
                             help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?',
                             help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true",
                       help="Deactivate the user interface and play as fast as possible.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?',
                         help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train_genetic and args.my_agent != "genetic_agent":
            raise ValueError("You can only train a genetic agent")

        if args.my_agent:
            if args.train_genetic:
                # set 4 players as genetic agent
                args.agents = [args.my_agent] * (s.MAX_AGENTS)
            else:
                # set 3 players as rule based agent and 1 chosen agent
                agents.append((args.my_agent, len(agents) < args.train))
                args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames

    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None

    def eval_genomes(genomes, config):
        global GENERATION
        print("Generation: ", GENERATION)
        if GENERATION % 2 == 0:
            args.scenario = "classic"
        else:
            args.scenario = "coin-heaven"
        GENERATION += 1

        # each generation the world is reset - check if it is possible sinse the world-class
        world = BombeRLeWorld(args, agents)
        if has_gui:
            gui = GUI(world)
        else:
            gui = None


        i = 0
        while i < len(genomes):
            # creates 4 agents at a time and runs them
            for index, elem in enumerate(genomes[i:i + 4]):
                genome_id = elem[0]
                genome = elem[1]

                genome.fitness = 0
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                world.agents[index].genetic_agent_net = net
                world.agents[index].train_genetic = True
                world.agents[index].genome = genome

            # execute the world
            world_controller(world, args.n_rounds, skip_end_round=False,
                             gui=gui, every_step=every_step, turn_based=args.turn_based,
                             make_video=args.make_video, update_interval=args.update_interval)

            # collect ge and fitness
            for g, agent in zip(genomes[i:i + 4], world.agents):
                g[1].fitness = agent.genome.fitness
                print(f"agent {agent.name} fitness: ", agent.genome.fitness)

            # get scores of all agents - no cosi prende lo score dell'ultimo round, deve prendere lo score di tutta la partita
            scores = [agent.total_score for agent in world.agents]
            print("scores: ", scores)
            for agent in world.agents:
                output = agent.genetic_agent_net.activate(scores)
                print(f"agent {agent.name} output: ", output)
                agent.weights = output

            i += 4

    if args.train_genetic:
        config_file = './agent_code/genetic_agent/config-feedforward.txt'
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        # p.add_reporter(neat.Checkpointer(5))

        # Run for up to 10 generations.
        winner = p.run(eval_genomes, 10)

        # save best genome
        with open('./agent_code/genetic_agent/winner.pkl', 'wb') as output:
            pickle.dump(winner, output)

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))
    else:
        if args.my_agent == "genetic_agent":
            # load winner network
            with open("./agent_code/genetic_agent/winner.pkl", "rb") as f:
                winner = pickle.load(f)

            config_file = './agent_code/genetic_agent/config-feedforward.txt'
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            world.agents[0].genetic_agent_net = winner_net

        world_controller(world, args.n_rounds, skip_end_round=True,
                         gui=gui, every_step=every_step, turn_based=args.turn_based,
                         make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()
