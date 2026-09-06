package exe.edu.cspg.common;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Settlement diagnostic B: the scheduling interval is off (0, legacy) unless configured. */
class DatacenterConfigSchedulingIntervalTest {

    @Test
    void defaultIsLegacyZeroAndBuilderSetsIt() {
        DatacenterConfig legacy = DatacenterConfig.builder().datacenterId(0).datacenterName("a").build();
        assertEquals(0.0, legacy.getSchedulingInterval(), 0.0);
        DatacenterConfig cert = DatacenterConfig.builder().datacenterId(0).datacenterName("a").schedulingInterval(1.0).build();
        assertEquals(1.0, cert.getSchedulingInterval(), 0.0);
    }
}
