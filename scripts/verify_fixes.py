
import os
import sys
import logging
import shutil

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import scripts to test
# We import them as modules so we can monkeypatch their global variables
import scripts.run_embc_fix as runner
import scripts.run_swap_test as swapper

def verify():
    log.info("STARTING VERIFICATION DRY RUN")
    
    # 1. Setup Test Output Directory
    TEST_OUT_DIR = os.path.abspath("outputs/verification_test")
    if os.path.exists(TEST_OUT_DIR):
        shutil.rmtree(TEST_OUT_DIR)
    os.makedirs(TEST_OUT_DIR)
    
    # 2. Patch run_embc_fix
    log.info("--- Patching run_embc_fix ---")
    runner.BASE_OUT_DIR = TEST_OUT_DIR
    runner.RESULTS_FILE = os.path.join(TEST_OUT_DIR, "embc_fix_raw_results.csv")
    runner.SUMMARY_FILE = os.path.join(TEST_OUT_DIR, "embc_fix_summary.csv")
    
    # Tiny Grid
    runner.SEEDS = [0] # Single seed
    runner.METHODS = ['erm'] # Single method
    runner.DIRECTIONS = [('ptbxl', 'chapman')] # Single direction
    runner.RHOS = [0.9] # Single SAST level
    runner.CONDITIONS = ['Clean', 'SAST_0.9'] # Min required for Swap Test (Clean + Poisoned)
    runner.EPOCHS = 1 # 1 Epoch training
    
    # Run
    log.info("Running Training/Evaluation Grid...")
    try:
        runner.main()
        log.info("✅ run_embc_fix completed successfully.")
    except Exception as e:
        log.error(f"❌ run_embc_fix failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Patch run_swap_test
    log.info("--- Patching run_swap_test ---")
    swapper.BASE_OUT_DIR = TEST_OUT_DIR
    swapper.SWAP_RESULTS_FILE = os.path.join(TEST_OUT_DIR, "swap_test_results.csv")
    swapper.SWAP_SUMMARY_FILE = os.path.join(TEST_OUT_DIR, "swap_test_summary.csv")
    
    swapper.SEEDS = [0]
    swapper.METHODS = ['erm']
    swapper.DIRECTIONS = [('ptbxl', 'chapman')]
    swapper.POISON_COND = "SAST_0.9"
    
    # Run
    log.info("Running Swap Test...")
    try:
        swapper.main()
        log.info("✅ run_swap_test completed successfully.")
    except Exception as e:
        log.error(f"❌ run_swap_test failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    log.info("VERIFICATION SUCCESSFUL: All scripts ran without error.")
    log.info(f"Test outputs can be found in {TEST_OUT_DIR}")

if __name__ == "__main__":
    verify()
