import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------
# 1. Data Reading and Random Sampling
data = pd.read_csv("a-N.csv", header=None, names=["a", "C", "M", "rho", "N"])

C_min, C_max = data["C"].min(), data["C"].max()
M_min, M_max = data["M"].min(), data["M"].max()
rho_min, rho_max = data["rho"].min(), data["rho"].max()

np.random.seed(114514)

num_samples = 50

C_samples = np.random.uniform(C_min, C_max, size=num_samples)
M_samples = np.random.uniform(M_min, M_max, size=num_samples)
rho_samples = np.random.uniform(rho_min, rho_max, size=num_samples)

samples_df = pd.DataFrame({
    "C": C_samples,
    "M": M_samples,
    "rho": rho_samples
})


def format_row(row):
    return f"{row['C']:.2e}\t{row['M']:.4f}\t{row['rho']:.4f}"


print(f"C : [{C_min:.2e}, {C_max:.2e}]")
print(f"M : [{M_min:.4f}, {M_max:.4f}]")
print(f"ρ : [{rho_min:.4f}, {rho_max:.4f}]")

plt.figure(figsize=(8, 6))
plt.scatter(M_samples, np.log10(C_samples), color='#8E7FB8', marker='o')
plt.xlabel('m', fontname="Times New Roman", fontweight="bold", fontsize=14)
plt.ylabel('log(C)', fontname="Times New Roman", fontweight="bold", fontsize=14)
ax = plt.gca()

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.grid(False)

plt.show()


# ---------------------------
# 2. Calculate FCG curve data
def calculate_delta_K(a_array, delta_P, B, W):


    a_avg_list = [(a_array[i] + a_array[i + 1]) / 2 for i in range(len(a_array) - 1)]

    alpha_list = [(a_avg + 10) / W for a_avg in a_avg_list]

    delta_K_list = []
    for alpha in alpha_list:
        poly_coef = 0.886 + 4.64 * alpha - 13.32 * alpha ** 2 + 14.72 * alpha ** 3 - 5.6 * alpha ** 4
        delta_K = (delta_P * (2 + alpha)) * poly_coef * 0.0316 / (B * (W ** 0.5) * ((1 - alpha) ** (3 / 2)))
        delta_K_list.append(delta_K)
    return a_avg_list, delta_K_list


a_array = np.linspace(0, 25, 31)
delta_P = 6300  # ΔP
B = 12  # B
W = 50  # W

a_avg_list, delta_K_list = calculate_delta_K(a_array, delta_P, B, W)

# ---------------------------
# 3. Generating the a–N curve using the FCG curve integral
# Formula: ΔN = Δa / (da/dN), where da/dN = C*(ΔK)^M
# For each sample, we compute the integral to obtain the boundary values for a (a_boundaries) and N (N_boundaries),
# while also storing the midpoints of the intervals (a_intervals) and the corresponding da/dN values (da_dn).
AN_data = []

print("\n a–N ：")
for idx, row in samples_df.iterrows():
    C_val = row["C"]
    M_val = row["M"]
    rho_val = row["rho"]

    da_dn_vals = [C_val * (dK ** M_val) for dK in delta_K_list]

    delta_a = np.diff(a_array)

    delta_N = [da * 0.001 / da_dn if da_dn != 0 else np.inf for da, da_dn in zip(delta_a, da_dn_vals)]

    N_values = [0]
    for dN in delta_N:
        N_values.append(N_values[-1] + dN)

    a_intervals = np.array(a_avg_list)
    N_boundaries = np.array(N_values)
    AN_data.append({
        "a_boundaries": a_array.copy(),
        "N_boundaries": N_boundaries,
        "a_intervals": np.array(a_avg_list),
        "da_dn": np.array(da_dn_vals),
        "C": C_val,
        "M": M_val,
        "rho": rho_val
    })

# ---------------------------
#
#
#
summary_data = []
for idx, data_item in enumerate(AN_data):
    N_end = data_item["N_boundaries"][-1]
    rho_val = data_item["rho"]
    summary_data.append({
        "Sample": idx,
        "N_end": N_end,
        "Original_rho": rho_val
    })
summary_df = pd.DataFrame(summary_data)

sorted_rho = sorted(summary_df["Original_rho"].tolist())

sorted_idx = summary_df.sort_values(by="N_end", ascending=False).index
new_rho_assignment = [None] * len(summary_df)
for i, idx_val in enumerate(sorted_idx):
    new_rho_assignment[idx_val] = sorted_rho[i]
summary_df["New_rho"] = new_rho_assignment

samples_df["New_rho"] = summary_df.sort_values("Sample")["New_rho"].values
for idx, data_item in enumerate(AN_data):
    data_item["rho"] = samples_df.loc[idx, "New_rho"]

plt.figure(figsize=(10, 6))
for idx, row in samples_df.iterrows():
    C_val = row["C"]
    M_val = row["M"]
    new_rho_val = row["New_rho"]

    da_dn_vals = [C_val * (dK ** M_val) for dK in delta_K_list]
    plt.loglog(delta_K_list, da_dn_vals, marker="o")

plt.xlabel("ΔK (MPa·m^1/2)", fontname="Times New Roman", fontweight="bold", fontsize=20)
plt.ylabel("da/dN (m/cycle)", fontname="Times New Roman", fontweight="bold", fontsize=20)

ax = plt.gca()

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=18)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))

plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

plt.figure(figsize=(10, 6))
for data_item in AN_data:
    a_vals = data_item["a_boundaries"]
    N_vals = data_item["N_boundaries"]
    plt.plot(N_vals, a_vals, marker="o")

plt.xlabel("N (cycles)", fontname="Times New Roman", fontweight="bold", fontsize=20)
plt.ylabel("a (mm)", fontname="Times New Roman", fontweight="bold", fontsize=20)

ax = plt.gca()

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0f}"))

plt.grid(False)

plt.show()

all_rows = []
for sample_idx, data_item in enumerate(AN_data):
    C_val = data_item["C"]
    M_val = data_item["M"]
    rho_val = data_item["rho"]

    a_int = data_item["a_intervals"]
    da_dn = data_item["da_dn"]

    N_bound = data_item["N_boundaries"]
    N_avg = (N_bound[:-1] + N_bound[1:]) / 2

    for i in range(len(a_int)):
        all_rows.append({
            "Sample": sample_idx + 1,
            "a": a_int[i],
            "C": C_val,
            "M": M_val,
            "rho": rho_val,
            "N": N_avg[i],
            "da/dN": da_dn[i],
            "Delta_K": delta_K_list[i]
        })

output_df = pd.DataFrame(all_rows, columns=["a", "C", "M", "rho", "N"])
output_csv_filename = "MC-expand data.csv"
output_df.to_csv(output_csv_filename, index=False)
print(f"\n: {output_csv_filename}")

# ---------------------------

output_df = pd.DataFrame(all_rows, columns=["M", "C"])
output_csv_filename = "C and M data.csv"
output_df.to_csv(output_csv_filename, index=False)
print(f"\n: {output_csv_filename}")

# ---------------------------

plt.figure(figsize=(10, 6))
plt.hist(summary_df["N_end"], bins=10, edgecolor='black', color='skyblue')

plt.xlabel("N_end (cycle)", fontname="Times New Roman", fontweight="bold", fontsize=14)
plt.ylabel("Frequency", fontname="Times New Roman", fontweight="bold", fontsize=14)

ax = plt.gca()

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.grid(False)

plt.show()

N_end = summary_df["N_end"].values
total = len(N_end)

bins = 10
counts, bin_edges = np.histogram(N_end, bins=bins)
percentages = counts / total * 100

hist_pct_df = pd.DataFrame({
    'BinStart': bin_edges[:-1],
    'BinEnd': bin_edges[1:],
    'Percentage': percentages
})
hist_pct_df.to_csv('Nend_histogram_percentage.csv', index=False)
print("Nend_histogram_percentage.csv")
print(hist_pct_df)

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), edgecolor='black')
plt.xlabel("N_end (cycles)", fontname="Times New Roman", fontweight="bold", fontsize=14)
plt.ylabel("Percentage (%)", fontname="Times New Roman", fontweight="bold", fontsize=14)
plt.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(summary_df["New_rho"], summary_df["N_end"], color='#8E7FB8', marker='o')
plt.xlabel("ρ (%)", fontname="Times New Roman", fontweight="bold", fontsize=14)
plt.ylabel("Nend (cycles)", fontname="Times New Roman", fontweight="bold", fontsize=14)

plt.grid(True, which='both', ls="--", lw=0.5)

ax = plt.gca()
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2)
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.grid(False)
plt.show()

scatter_df = summary_df[['New_rho', 'N_end']]

scatter_df.to_csv('rho_vs_Nend.csv', index=False)
print("rho_vs_Nend.csv")
