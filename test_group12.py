"""Reproducible tests for the Group 12 agent.

The normal scenario commands keep the checks used while developing the agent.
The extra suites add staff-style random streams, fixed-power games and matched
ablation tests.  S3-proxy is only a local stress test, not the official S3.
"""
import argparse
import contextlib
import importlib
import io
import json
import random
import statistics
import time

from diplomacy import Game

from agent_baselines import Agent, AttitudeAgent, GreedyAgent, RandomAgent, StaticAgent
from agent_group12 import OpponentModel, Search, StudentAgent
from game import copy_game


POWERS = list(Game().powers)
MIXED_POOL = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]


class HiddenProxy(Agent):
    """A per-game mixture for local S3 testing."""

    def __init__(self):
        super().__init__('S3 proxy')

    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.mode = random.choice(('greedy', 'attitude', 'noisy_greedy'))
        self.delegate = AttitudeAgent() if self.mode == 'attitude' else GreedyAgent()
        self.random_agent = RandomAgent()
        self.delegate.new_game(game, power_name)
        self.random_agent.new_game(game, power_name)

    def get_actions(self):
        if self.mode == 'noisy_greedy' and random.random() < 0.1:
            return self.random_agent.get_actions()
        return self.delegate.get_actions()

    def update_game(self, all_power_orders):
        self.delegate.update_game(all_power_orders)


class DisabledSearch(Search):
    """Used to measure the fallback greedy policy on its own."""

    def search(self, game, power_name, deadline):
        self.last_stats = {'disabled': 1}
        return None


class GreedyOnlyAgent(StudentAgent):
    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.search = DisabledSearch(self.evaluator, self.order_pruner)


class BeamOnlySearch(Search):
    """Keep the ordinary beam, but remove attack and convoy packages."""

    def _attack_combinations(self, incumbent, raw, game, power_name, deadline):
        if False:
            yield None

    def _convoy_combinations(self, incumbent, protected, raw, game, power_name, deadline):
        if False:
            yield None


class BeamOnlyAgent(StudentAgent):
    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.search = BeamOnlySearch(self.evaluator, self.order_pruner)


class HoldsOnlyOpponentModel(OpponentModel):
    """A weak forecast baseline that assumes every movement unit holds."""

    def predict(self, game, opponent_name, my_power_name):
        key = (id(game), game.get_current_phase(), game.get_hash(), my_power_name)
        if self._phase_key != key:
            self.begin_phase()
            self._phase_key = key
        if opponent_name in self.cache:
            return list(self.cache[opponent_name])
        self.calls += 1
        candidates = self._orders(game, opponent_name)
        orders = []
        if game.phase_type == 'M':
            for choices in candidates.values():
                hold = next((order for order in choices if order.endswith(' H')), None)
                if hold:
                    orders.append(hold)
        self.cache[opponent_name] = orders
        self.kinds[opponent_name] = 'holds-only'
        return list(orders)


class NoForecastAgent(StudentAgent):
    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.opponent_model = HoldsOnlyOpponentModel(self.memory, self)


class BasicRolloutAgent(StudentAgent):
    """One-step rollout without joint search or learned opponent types."""

    def new_game(self, game, power_name):
        super().new_game(game, power_name)
        self.search = DisabledSearch(self.evaluator, self.order_pruner)
        self.opponent_model = HoldsOnlyOpponentModel(self.memory, self)


ABLATIONS = [
    ('basic-rollout', BasicRolloutAgent),
    ('rollout+forecast', GreedyOnlyAgent),
    ('beam+forecast', BeamOnlyAgent),
    ('full+hold-forecast', NoForecastAgent),
    ('full-agent', StudentAgent),
]


def load_class(path):
    module_name, class_name = path.split(':', 1)
    return getattr(importlib.import_module(module_name), class_name)


def make_lineup(scenario, mine, rng, hidden_class, agent_class, staff_style):
    others = [power for power in POWERS if power != mine]
    hidden = rng.choice(others) if scenario.startswith('s3') else None
    classes = {}
    for power in POWERS:
        if power == mine:
            classes[power] = agent_class
        elif scenario == 's1':
            classes[power] = StaticAgent
        elif scenario == 'greedy':
            classes[power] = GreedyAgent
        elif power == hidden:
            classes[power] = hidden_class or HiddenProxy
        else:
            chooser = random if staff_style else rng
            classes[power] = chooser.choice(MIXED_POOL)
    return classes, hidden


def score_status(centres):
    if centres >= 18:
        return 'win'
    if centres == 0:
        return 'defeat'
    return 'survive'


def experiment(scenario, games, seed, trace=False, hidden_class=None,
               start_index=0, quiet=False, staff_style=False,
               fixed_power=None, agent_class=StudentAgent, label=None):
    """Run one scenario and return a JSON-serialisable summary."""
    if staff_style:
        random.seed(seed)

    scores = []
    raw_scores = []
    durations = []
    statuses = []
    model_matches = predicted_orders = prediction_calls = 0
    own_voids = baseline_errors = exceptions = 0
    proxy_modes = {}
    decision_routes = {}
    movement_fallbacks = {'greedy': 0, 'safe': 0}
    attack_evaluations = movement_decisions = 0
    convoy_attempts = convoy_successes = 0
    worst_decision = {}
    game_results = []
    per_power = {
        power: {'games': 0, 'wins': 0, 'survives': 0, 'defeats': 0, 'scores': []}
        for power in POWERS
    }

    for local_index in range(games):
        index = start_index + local_index
        if not staff_style:
            random.seed(seed + index)
        rng = random if staff_style else random.Random(seed + index)
        mine = fixed_power or POWERS[index % len(POWERS)]
        classes, hidden = make_lineup(
            scenario, mine, rng, hidden_class, agent_class, staff_style)
        agents = {power: cls() for power, cls in classes.items()}
        game = Game()
        for power, agent in agents.items():
            agent.new_game(copy_game(game), power)

        if scenario == 's3-proxy':
            mode = agents[hidden].mode
            proxy_modes[mode] = proxy_modes.get(mode, 0) + 1

        game_durations = []
        game_routes = {}
        game_convoys = game_convoy_successes = 0
        while not game.is_game_done and int(game.get_current_phase()[1:5]) < 1920:
            phase = game.get_current_phase()
            all_orders = {}
            for power, agent in agents.items():
                before = agent.game.get_state() if power == mine else None
                started = time.perf_counter()
                all_orders[power] = agent.get_actions()
                elapsed = time.perf_counter() - started
                if power != mine:
                    continue

                durations.append(elapsed)
                game_durations.append(elapsed)
                decision_phase = agent.last_decision.get('phase', game.phase_type)
                route = agent.last_decision.get('route', 'unknown')
                route_key = decision_phase + ':' + route
                decision_routes[route_key] = decision_routes.get(route_key, 0) + 1
                game_routes[route_key] = game_routes.get(route_key, 0) + 1
                if not worst_decision or elapsed * 1000 > worst_decision['ms']:
                    worst_decision = {
                        'game': index, 'power': mine, 'phase': phase,
                        'route': route, 'ms': elapsed * 1000,
                    }
                if decision_phase == 'M':
                    movement_decisions += 1
                    attack_evaluations += agent.search.last_stats.get('attack_evaluated', 0)
                    if route in movement_fallbacks:
                        movement_fallbacks[route] += 1

                assert elapsed < 0.9, ('deadline', phase, elapsed)
                after = agent.game.get_state()
                before.pop('timestamp', None)
                after.pop('timestamp', None)
                assert before == after, ('mutated game', phase)
                assert agent._valid(all_orders[power], game.phase_type, agent._candidates()), (
                    'invalid order list', phase, all_orders[power])
                prediction_calls += agent.opponent_model.calls
                assert agent.opponent_model.calls <= 6, ('prediction calls', phase)

            predictions = agents[mine]._opponent_predictions
            if phase.endswith('M'):
                for power in POWERS:
                    if power != mine and classes[power] is GreedyAgent:
                        actual = set(all_orders[power])
                        expected = set(predictions.get(power, []))
                        model_matches += len(expected & actual)
                        predicted_orders += len(actual)

            for power in POWERS:
                game.set_orders(power, all_orders[power])
                if power == mine:
                    assert not game.error, ('our submission', phase, game.error)
                else:
                    baseline_errors += len(game.error)
                game.error = []

            with contextlib.redirect_stdout(io.StringIO()):
                result = game.process()
                for agent in agents.values():
                    agent.update_game(all_orders)

            old_units = game.state_history.last_value()['units'][mine]
            for unit in old_units:
                own_voids += sum(
                    str(item) == 'void'
                    for item in result.results.get(unit.replace('*', ''), []))
            for order in all_orders[mine]:
                if not order.endswith(' VIA'):
                    continue
                words = order.split()
                successful = (
                    'A ' + words[3] in game.powers[mine].units
                    and not result.results.get(' '.join(words[:2]), []))
                convoy_attempts += 1
                convoy_successes += int(successful)
                game_convoys += 1
                game_convoy_successes += int(successful)

        raw_sc = len(game.powers[mine].centers)
        sc = min(18, raw_sc)
        status = score_status(sc)
        scores.append(sc)
        raw_scores.append(raw_sc)
        statuses.append(status)
        stats = per_power[mine]
        stats['games'] += 1
        stats[status + 's'] += 1
        stats['scores'].append(sc)
        exceptions += agents[mine].fallback_counts['exceptions']

        game_result = {
            'index': index,
            'seed': None if staff_style else seed + index,
            'power': mine,
            'sc': sc,
            'raw_sc': raw_sc,
            'status': status,
            'final_phase': game.get_current_phase(),
            'max_ms': max(game_durations, default=0.0) * 1000,
            'decision_routes': game_routes,
            'convoy_attempts': game_convoys,
            'convoy_successes': game_convoy_successes,
            'fallbacks': dict(agents[mine].fallback_counts),
            'opponents': {
                power: cls.__name__ for power, cls in classes.items() if power != mine
            },
        }
        game_results.append(game_result)
        if trace and not quiet:
            print(
                f"[{local_index + 1:>3}/{games}] {mine:<7} "
                f"SC={sc:>2} {status.upper():<7} "
                f"max={game_result['max_ms']:.1f} ms"
            )

    for stats in per_power.values():
        values = stats.pop('scores')
        count = max(1, stats['games'])
        stats['mean_sc'] = statistics.mean(values) if values else None
        stats['std_sc'] = statistics.pstdev(values) if values else None
        stats['win_rate'] = stats['wins'] / count
        stats['survival_rate'] = stats['survives'] / count
        stats['defeat_rate'] = stats['defeats'] / count

    ordered = sorted(durations)
    p99_index = min(len(ordered) - 1, int(len(ordered) * 0.99))
    summary = {
        'label': label or agent_class.__name__,
        'scenario': scenario,
        'games': games,
        'seed': seed,
        'start_index': start_index,
        'staff_style': staff_style,
        'fixed_power': fixed_power,
        'mean_sc': statistics.mean(scores),
        'std_sc': statistics.pstdev(scores),
        'raw_mean_sc': statistics.mean(raw_scores),
        'win_rate': statuses.count('win') / games,
        'survival_rate': statuses.count('survive') / games,
        'defeat_rate': statuses.count('defeat') / games,
        'max_ms': max(durations) * 1000,
        'p99_ms': ordered[p99_index] * 1000,
        'worst_decision': worst_decision,
        'greedy_exact_order_match': model_matches / max(1, predicted_orders),
        'prediction_calls': prediction_calls,
        'agent_exceptions': exceptions,
        'agent_void_results': own_voids,
        'baseline_input_errors': baseline_errors,
        'official_hidden': scenario == 's3',
        'per_power': per_power,
        'decision_routes': decision_routes,
        'movement_fallbacks': movement_fallbacks,
        'attack_evaluations': attack_evaluations,
        'attack_evaluations_per_movement': attack_evaluations / max(1, movement_decisions),
        'convoy_attempts': convoy_attempts,
        'convoy_successes': convoy_successes,
        'game_results': game_results,
    }
    if scenario == 's3-proxy':
        summary['note'] = 'Local proxy only; this is not the official hidden-agent result.'
        summary['proxy_modes'] = proxy_modes
    if not quiet:
        print_summary(summary)
    return summary


def print_summary(summary):
    name = summary['label']
    mode = 'staff stream' if summary['staff_style'] else 'matched seeds'
    print(f"\n{name} | {summary['scenario'].upper()} | games={summary['games']} | "
          f"seed={summary['seed']} | {mode}")
    print(f"{'Power':<10} {'Games':>5} {'SC mean±sd':>14} {'Win':>8} {'Survive':>9} {'Defeat':>8}")
    for power, stats in summary['per_power'].items():
        if not stats['games']:
            continue
        spread = f"{stats['mean_sc']:.2f}±{stats['std_sc']:.2f}"
        print(f"{power:<10} {stats['games']:>5} {spread:>14} "
              f"{stats['win_rate']:>7.1%} {stats['survival_rate']:>8.1%} "
              f"{stats['defeat_rate']:>7.1%}")
    all_spread = f"{summary['mean_sc']:.2f}±{summary['std_sc']:.2f}"
    print(f"{'ALL':<10} {summary['games']:>5} {all_spread:>14} "
          f"{summary['win_rate']:>7.1%} {summary['survival_rate']:>8.1%} "
          f"{summary['defeat_rate']:>7.1%}")
    print(f"Max / P99 decision: {summary['max_ms']:.2f} / {summary['p99_ms']:.2f} ms")
    print('Agent exceptions / invalid submissions / adjudicated voids: '
          f"{summary['agent_exceptions']} / 0 / {summary['agent_void_results']}")
    print('Movement fallback (greedy / safe): '
          f"{summary['movement_fallbacks']['greedy']} / "
          f"{summary['movement_fallbacks']['safe']}")
    if 'note' in summary:
        print(summary['note'])


def run_ablation(games, seed, scenario, quiet=False, trace=False, staff_style=False):
    results = []
    for label, agent_class in ABLATIONS:
        results.append(experiment(
            scenario, games, seed, quiet=quiet, trace=trace,
            staff_style=staff_style, agent_class=agent_class, label=label))
    if not quiet:
        print('\nAblation comparison (same powers, seeds and opponent lineups)')
        print(f"{'Version':<18} {'Mean SC':>8} {'Win':>8} {'Defeat':>8} {'Max ms':>9}")
        for result in results:
            print(f"{result['label']:<18} {result['mean_sc']:>8.2f} "
                  f"{result['win_rate']:>7.1%} {result['defeat_rate']:>7.1%} "
                  f"{result['max_ms']:>9.2f}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['scenario', 'smoke', 'final', 'ablation'],
                        default='scenario')
    parser.add_argument('--scenario', choices=['s1', 's2', 'greedy', 's3-proxy', 's3'],
                        default='s2')
    parser.add_argument('--games', type=int, default=50,
                        help='Games per scenario or per ablation version')
    parser.add_argument('--seed', type=int, default=2400)
    parser.add_argument('--start-index', type=int, default=0,
                        help='Continue deterministic seed and power rotation here')
    parser.add_argument('--power', choices=POWERS,
                        help='Test only this power (useful with --scenario greedy)')
    parser.add_argument('--staff-style', action='store_true',
                        help='Use one shared random stream, like the supplied test.py')
    parser.add_argument('--trace', action='store_true', help='Print one line per game')
    parser.add_argument('--quiet', action='store_true', help='Suppress tables and trace output')
    parser.add_argument('--hidden', help='Official hidden agent import path module:Class')
    parser.add_argument('--json', metavar='PATH', help='Write the complete results as JSON')
    args = parser.parse_args()
    if args.games <= 0 or args.start_index < 0:
        parser.error('--games must be positive and --start-index must be nonnegative')
    if args.scenario == 's3' and not args.hidden:
        parser.error('Official S3 requires --hidden module:Class; otherwise use s3-proxy')
    return parser, args


def main():
    parser, args = parse_args()
    hidden_class = load_class(args.hidden) if args.hidden else None
    common = {
        'seed': args.seed,
        'trace': args.trace,
        'quiet': args.quiet,
        'staff_style': args.staff_style,
    }

    if args.suite == 'scenario':
        results = experiment(
            args.scenario, args.games, hidden_class=hidden_class,
            start_index=args.start_index, fixed_power=args.power, **common)
    elif args.suite == 'ablation':
        if args.scenario == 's3':
            parser.error('Use --suite scenario for the official hidden-agent test')
        results = run_ablation(args.games, scenario=args.scenario, **common)
    else:
        scenarios = ['s1', 's2', 'greedy']
        games = 7 if args.suite == 'smoke' else args.games
        if args.suite == 'final':
            scenarios.append('s3-proxy')
        results = [
            experiment(scenario, games, fixed_power=args.power, **common)
            for scenario in scenarios
        ]

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
