# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Marko Lela (ORCID: 0009-0008-0768-5184)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, sys

if len(sys.argv) != 3:
    raise SystemExit("usage: extract_paper_numbers.py <summary.json> <null_summary.json>")

s = json.load(open(sys.argv[1], "r", encoding="utf-8"))
n = json.load(open(sys.argv[2], "r", encoding="utf-8"))

def g(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

theta_data = g(s, "theta", "theta_data")
pp_data_pct = g(s, "plateau_pct", "pp_data_pct")
s_levels = g(s, "s_levels", "s", "scales")

null_seed = g(n, "null_seed")
null_reps = g(n, "null_reps_per_patch", "null_reps")
theta_ref = g(n, "theta_null_ref_median_over_reps", "theta_null_ref")
pp_ref_frac = g(n, "pp_null_ref_median_over_reps", "pp_null_ref")  # often fraction 0..1

dtheta_med = g(n, "dtheta_med")
dtheta_ci95 = g(n, "dtheta_ci95")
dpp_med = g(n, "dpp_med")
dpp_ci95 = g(n, "dpp_ci95")

pp_ref_pct = None
if pp_ref_frac is not None:
    pp_ref_pct = 100.0 * float(pp_ref_frac)

print("SUMMARY")
print("  s_levels:", s_levels)
print("  theta_data:", theta_data)
print("  pp_data_pct:", pp_data_pct)
print("NULL")
print("  null_seed:", null_seed)
print("  null_reps_per_patch:", null_reps)
print("  theta_null_ref:", theta_ref)
print("  pp_null_ref_pct:", pp_ref_pct)
print("EFFECTS (paper)")
print("  dtheta_med:", dtheta_med)
print("  dtheta_ci95:", dtheta_ci95)
print("  dpp_med (pp):", dpp_med)
print("  dpp_ci95 (pp):", dpp_ci95)