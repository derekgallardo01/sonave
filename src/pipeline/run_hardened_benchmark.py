import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from src.pipeline.model_architectures import MultiFoundationAcousticEnsemble
from src.pipeline.hardened_benchmark import AdversarialBenchmarkEngine

def main():
    model = MultiFoundationAcousticEnsemble()
    engine = AdversarialBenchmarkEngine(num_samples=1000)
    results = engine.run_hardened_evaluation(model, device="cpu")

    print("\n====================================================")
    print(">>> ADVERSARIAL HARDENED BENCHMARK (1,000 SAMPLES)")
    print("====================================================")
    print("Total Samples Evaluated:             ", results["total_adversarial_samples"])
    print("Hardened Equal Error Rate (EER):      ", f"{results['hardened_equal_error_rate_pct']}%")
    print("Catch Rate @ 1% FAR:                 ", f"{results['catch_rate_at_1pct_far']}%")
    print("Catch Rate @ 5% FAR:                 ", f"{results['catch_rate_at_5pct_far']}%")
    print("Real Voice Preservation Accuracy:    ", f"{results['real_voice_preservation_accuracy_pct']}%")
    print("Hard In-The-Wild Deepfake Catch Rate: ", f"{results['in_the_wild_adversarial_catch_pct']}%")
    print("Operating Decision Threshold:        ", results["operating_threshold"])
    print("====================================================")
    print("Disaggregated Breakdown Across Voice Generators:")
    for gen, data in results["generator_catch_matrix"].items():
        print(f"  * {gen:<24}: Catch {data['catch_rate_pct']:>6.2f}%  (Mean Fake Score: {data['mean_fake_score']:.4f})")
    print("====================================================\n")

if __name__ == "__main__":
    main()
