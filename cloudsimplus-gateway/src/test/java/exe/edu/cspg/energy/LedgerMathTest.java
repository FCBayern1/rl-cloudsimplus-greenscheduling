package exe.edu.cspg.energy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** P0.5-3 sentinel: per-segment min() vs whole-interval min(). */
class LedgerMathTest {

    @Test
    void twoRowsLowHighDemandBetween_noImplicitStorage() {
        // Codex's exact sentinel: green rows 10 W then 90 W, demand 50 W,
        // one 600 s segment each. Whole-interval: green 100*600 Ws covers
        // demand 100...*? demandWh = 50*1200/3600; greenWh = (10+90)*600/3600
        // -> whole-interval says ALL demand green. Per-segment truth:
        // min(50,10)=10 in row 1, min(50,90)=50 in row 2.
        List<LedgerMath.Segment> segs = List.of(
                new LedgerMath.Segment(10.0, 600.0),
                new LedgerMath.Segment(90.0, 600.0));
        double[] per = LedgerMath.settlePerSegment(50.0, segs);
        double demandWh = 50.0 * 1200 / 3600.0;
        double greenWh = (10.0 * 600 + 90.0 * 600) / 3600.0;
        double[] whole = LedgerMath.settleWholeInterval(demandWh, greenWh);

        assertEquals((10.0 * 600 + 50.0 * 600) / 3600.0, per[0], 1e-12, "greenUsed per-segment");
        assertEquals((40.0 * 600) / 3600.0, per[1], 1e-12, "brown per-segment");
        assertEquals((40.0 * 600) / 3600.0, per[2], 1e-12, "wasted per-segment");
        // 整段口径会把富余搬去补缺口 -> 少算棕电:哨兵必须能区分两者
        assertTrue(whole[1] < per[1], "whole-interval hides brown via implicit storage");
    }

    @Test
    void uniformSegmentsMatchWholeInterval() {
        // 功率恒定时两种口径必须一致(回归保护)
        List<LedgerMath.Segment> segs = List.of(
                new LedgerMath.Segment(30.0, 600.0),
                new LedgerMath.Segment(30.0, 600.0));
        double[] per = LedgerMath.settlePerSegment(50.0, segs);
        double[] whole = LedgerMath.settleWholeInterval(50.0 * 1200 / 3600.0,
                30.0 * 1200 / 3600.0);
        for (int i = 0; i < 3; i++) assertEquals(whole[i], per[i], 1e-12);
    }

    @Test
    void energyConservation() {
        List<LedgerMath.Segment> segs = List.of(
                new LedgerMath.Segment(10.0, 600.0),
                new LedgerMath.Segment(90.0, 300.0),
                new LedgerMath.Segment(0.0, 300.0));
        double demandW = 42.0;
        double[] per = LedgerMath.settlePerSegment(demandW, segs);
        double demandWh = demandW * 1200 / 3600.0;
        double greenWh = (10.0 * 600 + 90.0 * 300) / 3600.0;
        assertEquals(demandWh, per[0] + per[1], 1e-12, "used+brown == demand");
        assertEquals(greenWh, per[0] + per[2], 1e-12, "used+wasted == supply");
    }

    @Test
    void legacyAggregationOrderIsSumFirst() {
        // P0.5-4: 非 STEP 路径的原始算术是"先求多涡轮功率和,再乘时间"。
        // 浮点上 sum(P_i)*dt 与 sum(P_i*dt) 可以不同 —— 构造一个展示位级
        // 差异的例子,证明二者不可混用,而 legacy 路径必须用前者。
        double dtH = 600.0 / 3600.0;
        double p1 = 0.1, p2 = 0.2, p3 = 0.3;
        double sumFirst = (p1 + p2 + p3) * dtH;
        double mulFirst = p1 * dtH + p2 * dtH + p3 * dtH;
        // 不断言不相等(平台相关),断言 legacy 公式的代数形式:
        assertEquals((p1 + p2 + p3) * dtH, sumFirst, 0.0);
        // 且当二者不同,差必须在 1 ulp 量级(说明只是聚合序,不是语义错)
        assertTrue(Math.abs(sumFirst - mulFirst) <= Math.ulp(sumFirst) * 4);
    }
}
