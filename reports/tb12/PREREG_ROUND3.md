# TB12 第三轮(最终)判决预注册

```json
{
 "title": "TB12 \u7b2c\u4e09\u8f6e(\u6700\u7ec8)\u5224\u51b3\u9884\u6ce8\u518c \u2014\u2014 \u63a5\u7ebf\u4fee\u590d\u540e,\u65e0\u8bba\u8fc7\u5426\u6536\u5175",
 "frozen_at": "2026-08-23 11:52",
 "verdict_dataset": {
  "turbines": [
   114,
   115
  ],
  "year": 2021,
  "sha256": {
   "Turbine_114_2021.csv": "3e04c2ae47dc7c36a234ad272760de491f6d437e0adba94776e2e1287d2d0502",
   "Turbine_115_2021.csv": "1cd710a81777c9e3c8432128d4650fd3f668a9c62d730d2ffc5039269621a52e"
  },
  "rows": 52559,
  "selection_rationale": "\u6c61\u67d3\u7684 110-113 \u4e4b\u540e\u6309\u7f16\u53f7\u7684\u4e0b\u4e00\u5bf9;\u4ed3\u5e93\u65e0\u65e2\u5f80\u5206\u6790\u8bb0\u5f55",
  "never_analyzed_before_this_freeze": true
 },
 "offsets": [
  480,
  1488,
  4024,
  4206,
  5072,
  5103,
  5637,
  5902,
  6559,
  7120,
  9876,
  9926,
  10139,
  10422,
  12863,
  13394,
  15897,
  16448,
  17722,
  17872,
  18255,
  18540,
  19438,
  20636,
  21380,
  21743,
  23240,
  23331,
  23608,
  26135,
  27267,
  27932,
  28217,
  28854,
  32653,
  32693,
  32982,
  33726,
  34836,
  35143,
  36565,
  37369,
  37595,
  37722,
  37754,
  40053,
  41083,
  42255,
  42274,
  42473,
  43014,
  43271,
  44730,
  44782,
  45856,
  46971,
  47773,
  48996,
  50435,
  51948
 ],
 "offsets_provenance": "\u6cbf\u7528 round-2 \u7684 60 \u4e2a\u5b63\u5ea6\u5206\u5c42\u504f\u79fb(seed 20260825)",
 "bootstrap": {
  "n": 10000,
  "seed": 20260827,
  "blocks": "\u5b63\u5ea6"
 },
 "jar_sha256": "12c30342f26e998982b6cbca5a8d836e6cef3e5c03fa0200aed635ac416bcf77",
 "runner_commit": "6f23432f2028708c203744bad8daf623bea61f26",
 "trace_sha256": "5a7737c0fc917f8793fcc4a0c93993bb4d775dfa5b1f2bb045ae6d00f35ac20f",
 "comparator": "greenfollow (step-6 \u51bb\u7ed3\u6cbf\u7528,\u4e0d\u91cd\u9009)",
 "arms": [
  "nowait",
  "greenfollow",
  "hazard",
  "dpcont",
  "clair"
 ],
 "gates": {
  "main": "pooled <= -5%",
  "strong": "pooled <= -8%",
  "direction": ">= 42/60 strict",
  "ci": "95% CI upper <= -5% (seed 20260827)",
  "validity": "60/60 finished 5/5, no truncation",
  "wiring": "planning == environment == prereg turbines(\u4e09\u65b9\u6838\u5bf9) + \u9010\u4f4d\u542f\u52a8\u54e8\u5175"
 },
 "diagnostic_context_T110_111_wired": {
  "pooled": -0.1497,
  "wins": 39,
  "ties": 10,
  "losses": 11,
  "median": -0.0669
 },
 "terminal": "\u672c\u8f6e\u4e3a\u6700\u7ec8\u8f6e,\u65e0\u8bba\u901a\u8fc7\u4e0e\u5426\u6536\u5175,\u4e0d\u8bbe\u7b2c\u56db\u8f6e"
}
```
