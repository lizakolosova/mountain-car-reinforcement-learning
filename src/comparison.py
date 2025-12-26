import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import glob
import os


def evaluate_model(model_path, episodes=20):
    model = SAC.load(model_path)
    env = gym.make("MountainCarContinuous-v0")

    returns = []
    lengths = []
    max_positions = []
    successes = []
    distance_to_goal = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        max_pos = -999

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            position = env.unwrapped.state[0]
            max_pos = max(max_pos, position)

        success = max_pos >= 0.45
        dist = abs(0.45 - max_pos)

        returns.append(ep_return)
        lengths.append(ep_length)
        max_positions.append(max_pos)
        successes.append(success)
        distance_to_goal.append(dist)

    env.close()

    return {
        'mean_max_pos': np.mean(max_positions),
        'std_max_pos': np.std(max_positions),
        'success_rate': np.mean(successes) * 100,
        'mean_distance': np.mean(distance_to_goal),
        'std_distance': np.std(distance_to_goal),
        'mean_length': np.mean(lengths),
        'best_position': np.max(max_positions)
    }


def comparison():
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')

    print("\n" + "=" * 80)
    print("BASELINE vs CUSTOM COMPARISON")
    print("=" * 80 + "\n")

    results = {'baseline': [], 'custom': []}

    for exp_type in ['baseline', 'custom']:
        print(f"{exp_type.upper()} Results:")
        print("-" * 80)

        for trial in range(3):
            checkpoint_dir = f"logs/{exp_type}/SAC_trial{trial}/checkpoints/"

            if not os.path.exists(checkpoint_dir):
                continue

            checkpoints = glob.glob(os.path.join(checkpoint_dir, "rl_model_*_steps.zip"))
            if not checkpoints:
                continue

            checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
            latest = checkpoints[-1]

            trial_results = evaluate_model(latest, episodes=20)
            trial_results['trial'] = trial
            results[exp_type].append(trial_results)

            print(f"  Trial {trial}: max_pos={trial_results['mean_max_pos']:.3f}, "
                  f"distance_to_goal={trial_results['mean_distance']:.3f}, "
                  f"success={trial_results['success_rate']:.0f}%")

        print()

    if not results['baseline'] or not results['custom']:
        print("Missing data")
        return

    baseline_pos = np.mean([r['mean_max_pos'] for r in results['baseline']])
    baseline_dist = np.mean([r['mean_distance'] for r in results['baseline']])
    baseline_success = np.mean([r['success_rate'] for r in results['baseline']])

    custom_pos = np.mean([r['mean_max_pos'] for r in results['custom']])
    custom_dist = np.mean([r['mean_distance'] for r in results['custom']])
    custom_success = np.mean([r['success_rate'] for r in results['custom']])

    print("=" * 80)
    print("COMPREHENSIVE COMPARISON (Task Performance)")
    print("=" * 80 + "\n")

    print(f"{'Metric':<30} {'Baseline':<20} {'Custom':<20} {'Improvement':<15}")
    print("-" * 85)

    print(f"{'Max Position Reached':<30} {baseline_pos:<20.3f} {custom_pos:<20.3f} ", end="")
    pos_improvement = custom_pos - baseline_pos
    print(f"{pos_improvement:+.3f}")

    print(f"{'Distance to Goal (0.45)':<30} {baseline_dist:<20.3f} {custom_dist:<20.3f} ", end="")
    dist_improvement = baseline_dist - custom_dist
    print(f"{dist_improvement:+.3f} ✓")

    pct_closed = (dist_improvement / baseline_dist) * 100
    print(f"{'% of Distance Closed':<30} {'':<20} {'':<20} {pct_closed:.1f}%")

    print(f"{'Success Rate':<30} {baseline_success:<20.1f}% {custom_success:<20.1f}% ", end="")
    if custom_success > baseline_success:
        print(f"+{custom_success - baseline_success:.1f}%")
    else:
        print("0%")

    print()

    print("=" * 80)
    print("KEY INSIGHTS FOR THE REPORT")
    print("=" * 80 + "\n")

    print(f"  • Baseline stuck at position {baseline_pos:.3f} (left valley)")
    print(f"  • {baseline_dist:.3f} units away from goal")

    print(f"  • Custom reward reaches position {custom_pos:.3f}")
    print(f"  • Only {custom_dist:.3f} units from goal")
    print(f"  • Closes {pct_closed:.1f}% of the distance to goal")
    print(
        f"  • Improvement: {pos_improvement:+.3f} position units ({abs(pos_improvement / baseline_pos) * 100:.0f}% better)\n")


    print("-" * 80)
    print(
        f"Baseline max position:     {baseline_pos:.3f} ± {np.std([r['mean_max_pos'] for r in results['baseline']]):.3f}")
    print(f"Custom max position:       {custom_pos:.3f} ± {np.std([r['mean_max_pos'] for r in results['custom']]):.3f}")
    print(f"Absolute improvement:      {pos_improvement:+.3f}")
    print(f"Relative improvement:      {abs(pos_improvement / baseline_pos) * 100:+.0f}%")
    print(f"Distance closed:           {pct_closed:.1f}%")
    print(f"Number of trials:          {len(results['baseline'])} (baseline), {len(results['custom'])} (custom)")
    print("-" * 80 + "\n")


def extension_comparison(base_dir="./logs/extension"):
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    if not os.path.exists(base_dir):
        print("No extension logs found.")
        return

    print("\n" + "=" * 80)
    print("CUSTOM REWARD + HYPERPARAMETER TUNING (EXTENSION)")
    print("=" * 80 + "\n")

    import glob
    import numpy as np

    # loop over learning_rate_0.0001, learning_rate_0.0003, ...
    for hp_setting in sorted(os.listdir(base_dir)):
        hp_path = os.path.join(base_dir, hp_setting)
        if not os.path.isdir(hp_path):
            continue

        trial_results = []
        print(f"Hyperparameter setting: {hp_setting}")
        print("-" * 80)

        for trial_dir in sorted(os.listdir(hp_path)):
            checkpoint_dir = os.path.join(hp_path, trial_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                continue

            checkpoints = glob.glob(os.path.join(checkpoint_dir, "rl_model_*_steps.zip"))
            if not checkpoints:
                continue

            checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
            latest = checkpoints[-1]

            res = evaluate_model(latest, episodes=20)
            trial_results.append(res)
            print(
                f" {trial_dir}: max_pos={res['mean_max_pos']:.3f}, "
                f"dist={res['mean_distance']:.3f}, success={res['success_rate']:.0f}%"
            )

        if not trial_results:
            print(" No valid trials.\n")
            continue

        mean_pos = np.mean([r["mean_max_pos"] for r in trial_results])
        mean_dist = np.mean([r["mean_distance"] for r in trial_results])
        mean_succ = np.mean([r["success_rate"] for r in trial_results])

        print(
            f" => mean max_pos={mean_pos:.3f}, "
            f"mean_dist={mean_dist:.3f}, "
            f"mean_success={mean_succ:.1f}%\n"
        )


if __name__ == "__main__":
    comparison()           # baseline vs custom
    extension_comparison() # custom vs tuned learning rates
