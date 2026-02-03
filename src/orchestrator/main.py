import argparse
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from src.training.worker import train_nas_candidate

def get_search_space():
    """
    Define the Hyperparameter Search Space.
    This includes both architectural parameters (NAS) and training hyperparameters.
    """
    return {
        # Architectural params
        "num_layers": tune.randint(2, 5), # Search between 2 and 4 layers
        "l0_kernel": tune.choice([3, 5]),
        "l0_out": tune.choice([16, 32, 64]),
        "l0_act": tune.choice(["relu", "silu"]),
        "l0_res": tune.choice([True, False]),
        
        "l1_kernel": tune.choice([3, 5]),
        "l1_out": tune.choice([32, 64, 128]),
        "l1_act": tune.choice(["relu", "silu"]),
        "l1_res": tune.choice([True, False]),

        "l2_kernel": tune.choice([3, 5]),
        "l2_out": tune.choice([64, 128, 256]),
        "l2_act": tune.choice(["relu", "silu"]),
        "l2_res": tune.choice([True, False]),

        # Training params
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64])
    }

def run_nas(args):
    """
    Orchestrate the NAS process.
    """
    # Initialize Ray (Connect to cluster or start local)
    if args.address:
        ray.init(address=args.address)
    else:
        # Limit resources for demo safety on local machine
        ray.init(num_cpus=4, num_gpus=1 if args.gpu else 0)

    print(f"🚀 Starting NAS with Strategy: {args.strategy}")
    
    # scheduler: Early Stopping
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=2,
        reduction_factor=2
    )
    
    # search_alg: The Strategy
    search_alg = None
    if args.strategy == "bayesian":
        # Uses Optuna internally
        search_alg = OptunaSearch()
    elif args.strategy == "random":
        pass # Default behavior
    
    tuner = tune.Tuner(
        train_nas_candidate,
        param_space=get_search_space(),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.trials,
            max_concurrent_trials=args.parallelism
        ),
        run_config=ray.train.RunConfig(
            name="nas_experiment",
            storage_path="./results",
            stop={"accuracy": 0.99} # Stop if perfect model found
        )
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result("accuracy", "max")
    print("\n🏆 Best Architecture Found:")
    print(f"  Accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"  Config: {best_result.config}")
    print(f"  Checkpoint: {best_result.checkpoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise NAS Orchestrator")
    parser.add_argument("--strategy", type=str, default="bayesian", choices=["random", "bayesian"], help="Search Strategy")
    parser.add_argument("--trials", type=int, default=10, help="Number of architecture candidates to evaluate")
    parser.add_argument("--parallelism", type=int, default=2, help="Number of concurrent trials")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--address", type=str, help="Ray Cluster Address (e.g. 'auto')")
    
    args = parser.parse_args()
    run_nas(args)
