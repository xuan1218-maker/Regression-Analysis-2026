from utils import setup_results_dir
from scenarios import scenario_A, scenario_B

def main():
    print("="*50)
    print("第六周实验：回归推断引擎")
    print("="*50)
    
    results_path = setup_results_dir()
    scenario_A(results_path)
    scenario_B(results_path)
    
    print(f"\n报告保存在: {results_path}/")

if __name__ == "__main__":
    main()