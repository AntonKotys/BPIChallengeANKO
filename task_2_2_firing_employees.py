import pandas as pd
df = pd.read_csv("sim_predicted_earliest_available.csv")

human_df = df[
    (df["org:resource"] != "SYSTEM_W") &
    (df["concept:name"] != "END")
]

usage = human_df["org:resource"].value_counts()

# the ones with the smallest busy time are the ones to remove
#  FIRED = {"User_85", "User_74"}
# then run the simulation_engine_core_V1_8 which excludes these employees
print(usage.sort_values())
#           strategy  avg_cycle_time_days  avg_activity_delay_hours  avg_human_resource_occupation_pct  human_resource_fairness_jain  human_delayed_share_pct
# earliest_available            18.820257                  8.920720                           2.035191                      0.171511                69.376518
# earliest_available            20.694892                  9.385391                           1.798552                      0.150345                69.507060




