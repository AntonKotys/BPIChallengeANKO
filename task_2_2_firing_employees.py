import pandas as pd

df = pd.read_csv("sim_outputs/sim_predicted_earliest_available_advanced_roles.csv")

name = df["concept:name"].astype(str).str.strip()

# exclude those who are W_, but still include those who have &A_ &or O_
human_df = df[
    (~name.str.startswith("W_", na=False)) &
    (name != "END")
]

usage = human_df["org:resource"].value_counts()
print(usage.sort_values())

# the ones with the smallest number of assigned tasks are the ones to remove
#  FIRED = {"User_85", "User_103"}
# then run the simulation_engine_core_V1_8 which excludes these employees
# print(usage.sort_values())


# strategy                                        avg_cycle_time_days  avg_activity_delay_hours        avg_resource_occupation_pct       resource_fairness_jain      delayed_share_pct
# earliest_available_advanced_roles before            21.887918                 9.511540                          78.438628                      0.621013                52.185871
# earliest_available_advanced_roles                   22.014988                 9.486734                          79.905268                      0.637195                44.637582


