1. Install Python Dependencies

```bash
# Using virtual environment
# Windows:
python -m venv venv
# macOS:
python3 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
# Windows:
pip install -r requirements.txt
# macOS:
pip3 install -r requirements.txt

# Run Simulation Engine
# Windows (for macOS use python3 instead of just python):
python .\simulation_engine_core_final_version.py    # main simulation engine; outputs are generated in the "sim_outputs" folder
python .\1.6_test.py                                # tests for simulation outputs; set the desired .csv file to test in line 6 of 1.6_test.py  
python .\1.6_grid_search.py                         # tests for finding optimal advanced role permissions

```

# Configs
```python
USE_ADVANCED_PERMISSIONS = True # Advanced role permissions; set to false for basic historical resource permissions
NUM_CASES = 1000                 # Total number of cases to generate
MAX_EVENTS_PER_CASE = 200       # Max events a single case can have
```

