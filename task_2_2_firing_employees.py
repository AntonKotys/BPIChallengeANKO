import pandas as pd
df = pd.read_csv("sim_outputs/sim_predicted_earliest_available.csv")

human_df = df[
    (df["org:resource"] != "SYSTEM_W") &
    (df["concept:name"] != "END")
]

usage = human_df["org:resource"].value_counts()

# the ones with the smallest busy time are the ones to remove
#  FIRED = {"User_85", "User_74"}
# then run the simulation_engine_core_V1_8 which excludes these employees
print(usage.sort_values())

#   strategy                                   avg_cycle_time_days     avg_activity_delay_hours      avg_human_resource_occupation_pct    human_resource_fairness_jain  human_delayed_share_pct
# earliest_available_advanced_roles before            20.661086                  8.983227                           1.622071                      0.222547                73.775452
# earliest_available_advanced_roles lowest busy time  20.651417                  9.146179                           1.157157                      0.236244                71.905266
# earliest_available_advanced_roles  brute force      20.291466                  9.231054                           3.342048                      0.138523                74.402090


# strategy                                        avg_cycle_time_days  avg_activity_delay_hours        avg_resource_occupation_pct       resource_fairness_jain      delayed_share_pct
# earliest_available_advanced_roles before            21.887918                  9.511540                          78.438628                      0.621013                52.185871
# earliest_available_advanced_roles lowest busy time  19.796923                  9.163847                          71.735726                      0.610781                50.553569
# earliest_available_advanced_roles  brute force      19.807291                  9.162721                          66.968307                      0.606767                50.868245


