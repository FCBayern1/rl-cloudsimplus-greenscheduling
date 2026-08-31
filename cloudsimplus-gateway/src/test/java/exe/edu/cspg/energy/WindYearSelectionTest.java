package exe.edu.cspg.energy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The wind year has to be selectable, and 2021 has to stay the default.
 *
 * The resolver listed Turbine_<id>_2021.csv first unconditionally, and that file always
 * exists, so no configuration could ever reach another year. The cross-year confirmation
 * set in the planner gate was therefore unrunnable, which only surfaced when it was time
 * to run it. Defaulting to 2021 keeps every existing run resolving to the same file.
 */
public class WindYearSelectionTest {

    @AfterEach
    public void restoreDefault() {
        GreenEnergyProvider.setPreferredWindYear(2021);
    }

    @Test
    public void theDefaultIsTheHistoricalYear() {
        assertEquals(2021, GreenEnergyProvider.getPreferredWindYear());
    }

    @Test
    public void thePreferenceIsHonoured() {
        GreenEnergyProvider.setPreferredWindYear(2020);
        assertEquals(2020, GreenEnergyProvider.getPreferredWindYear());
    }
}
