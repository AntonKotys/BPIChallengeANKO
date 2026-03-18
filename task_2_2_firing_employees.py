import pandas as pd
df = pd.read_csv("sim_outputs/sim_predicted_earliest_available_advanced_roles.csv")

human_df = df[
    (df["org:resource"] != "SYSTEM_W") &
    (df["concept:name"] != "END")
]

usage = human_df["org:resource"].value_counts()

# the ones with the smallest busy time are the ones to remove
#  FIRED = {"User_145", "User_147"}
# then run the simulation_engine_core_V1_8 which excludes these employees
print(usage.sort_values())


# strategy                                        avg_cycle_time_days  avg_activity_delay_hours        avg_resource_occupation_pct       resource_fairness_jain      delayed_share_pct
# earliest_available_advanced_roles before            21.887918                 9.511540                          78.438628                      0.621013                52.185871
# earliest_available_advanced_roles lowest busy time  19.803653                 9.234716                          63.290774                      0.604219                51.130704
# earliest_available_advanced_roles  brute force      19.807291                 9.162721                          66.968307                      0.606767                50.868245


