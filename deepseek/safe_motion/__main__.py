"""命令行入口：python -m safe_motion <scenario.json> [--plot] [--out 路径]。"""
import argparse
import json
import os

from .replay import load_scenario, print_report, run_scenario


def main():
    parser = argparse.ArgumentParser(description="SafeMotion 轨迹回放")
    parser.add_argument("scenario", help="场景 JSON 路径")
    parser.add_argument("--plot", action="store_true", help="生成 3D 可视化图")
    parser.add_argument("--out", default="artifacts", help="输出目录")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    report = run_scenario(scenario)
    print_report(report)

    if args.plot:
        from .visualize import plot_replay
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"{report['name']}.png".replace(" ", "_"))
        plot_replay(scenario, report, out_path)
        print(f"\n3D 图已保存: {out_path}")


if __name__ == "__main__":
    main()
