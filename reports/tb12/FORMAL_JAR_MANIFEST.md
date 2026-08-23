# TB12 第二轮正式 jar 清单（P0.5 后,2026-08-23 02:35 构建）

| 项 | 值 |
|---|---|
| jar SHA256(完整 64 位) | `8a3b5e1fab8d2ec1f00eecda2c780b65ccbb5b2bc982e995c3497f4b3a013cc4` |
| jar mtime | 2026-08-23 02:35:22 |
| 构建源 git commit | 7ec9c859a49216330db131a4b16162928085ba54（本清单与 P0.5 代码同一提交入库） |
| Java | openjdk version 21.0.11 2026-04-21 |
| JVM 时区来源 | gradle jvmargs -Duser.country=GB(Europe/London);DST 缺陷已由行语义绕开 |
| GATEWAY_LIBS | 本地一律 unset(gradlew 路径);若设必须指向 build/install/.../lib 且 SHA 与本清单一致 |

## 四项 sentinel

| sentinel | 结果 |
|---|---|
| 派发(VM free-PE) | ✅ c0/c1/c2→VM0/1/2 并行;完成提前 ~2000s |
| STEP 行语义 | ✅ 四层门:观测 float32 级 / bins 2.84e-14 W / 能量 0.000–0.001% |
| DST 删行 | ✅ off=26000(空洞 row 12389 之后)从 89.6W 错位 → 逐位 |
| loaded-ledger 逐段 min | ✅ LedgerMathTest 哨兵(10/90W 两行,需求 50W):区分整段/逐段口径,能量守恒 |

## P0.5 后门回归

CONSISTENCY GATE PASSED(consistency_gate6.log,与 P0.5 前逐位一致)
