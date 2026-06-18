package exe.edu.cspg.common;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the deferrable-jobs temporal lever fields on
 * {@link CloudletDescriptor} (2026-06-18, Phase 1).
 *
 * Guards: backward-compatible 4-arg ctor defaults to non-deferrable; the
 * 5-arg ctor carries the deadline; negative deadlines clamp to 0; the new
 * field participates in equals/hashCode. See docs/Deferrable_Jobs_Lever.md.
 */
public class CloudletDescriptorDeferrableTest {

    @Test
    void legacyConstructorIsNonDeferrable() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 1000, 2);
        assertEquals(0, d.getDeferDeadlineSteps());
        assertFalse(d.isDeferrable(), "4-arg ctor must default to non-deferrable");
    }

    @Test
    void fullConstructorCarriesDeadline() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 1000, 2, 12);
        assertEquals(12, d.getDeferDeadlineSteps());
        assertTrue(d.isDeferrable(), "deferDeadlineSteps>0 must be deferrable");
    }

    @Test
    void zeroDeadlineMeansNotDeferrable() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 1000, 2, 0);
        assertFalse(d.isDeferrable());
    }

    @Test
    void negativeDeadlineClampsToZero() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 1000, 2, -5);
        assertEquals(0, d.getDeferDeadlineSteps());
        assertFalse(d.isDeferrable());
    }

    @Test
    void deadlineParticipatesInEqualsAndHashCode() {
        CloudletDescriptor a = new CloudletDescriptor(1, 0, 1000, 2, 10);
        CloudletDescriptor b = new CloudletDescriptor(1, 0, 1000, 2, 10);
        CloudletDescriptor c = new CloudletDescriptor(1, 0, 1000, 2, 20);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c, "different deadline must not be equal");
    }

    @Test
    void legacyAndZeroDeadlineCtorsAreEquivalent() {
        assertEquals(new CloudletDescriptor(7, 3, 500, 1),
                     new CloudletDescriptor(7, 3, 500, 1, 0));
    }
}
