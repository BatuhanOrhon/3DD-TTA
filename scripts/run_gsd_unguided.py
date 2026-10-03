"""Full-suite diffusion without SCD or spectral guidance, seeds 0/1/2."""
from run_gsd_guidance_ablation import main

if __name__ == "__main__":
    raise SystemExit(main("unguided"))
