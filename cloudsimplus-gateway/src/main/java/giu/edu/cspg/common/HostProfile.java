package giu.edu.cspg.common;
import giu.edu.cspg.common.HostProfile;
import lombok.Getter;
/**
 * Represents a server hardware profile with specific power consumption characteristics.
 * Based on real server data from SPEC power_ssj2008 benchmark.
 */
@Getter
public class HostProfile {
    private final String name;
    private final int pes;                  // Number of CPU cores
    private final long peMips;              // MIPS per core
    private final long ram;                 // RAM in MB
    private final long bw;                  // Bandwidth in Mbps
    private final long storage;             // Storage in MB
    private final double maxPowerW;         // Peak power consumption in Watts
    private final double staticPowerPercent; // Idle power as percentage of max power

    /**
     * Creates a host profile with specified characteristics.
     *
     * @param name Server model name (e.g., "HP-DL360-G7")
     * @param pes Number of CPU cores
     * @param peMips MIPS per core
     * @param ram RAM in MB
     * @param bw Bandwidth in Mbps
     * @param storage Storage in MB
     * @param maxPowerW Maximum power consumption at 100% utilization (Watts)
     * @param staticPowerPercent Idle power as percentage of max power (0-100)
     */
    public HostProfile(String name, int pes, long peMips, long ram, long bw, long storage,
                       double maxPowerW, double staticPowerPercent) {
        this.name = name;
        this.pes = pes;
        this.peMips = peMips;
        this.ram = ram;
        this.bw = bw;
        this.storage = storage;
        this.maxPowerW = maxPowerW;
        this.staticPowerPercent = staticPowerPercent;
    }

    // Predefined profiles based on SPEC power_ssj2008 real server data

    /**
     * HP ProLiant DL360 G7 - Low power, efficient server
     * Idle: 58W, Peak: 208W (27.9% idle power)
     */
    public static HostProfile LOW_POWER() {
        return new HostProfile(
            "HP-DL360-G7-LowPower",
            8,          // 8 cores
            50000,      // 50000 MIPS per core
            32768,      // 32 GB RAM
            10000,      // 10 Gbps
            100000,     // 100 GB storage
            208,        // 208W peak power
            27.9        // 58W idle / 208W = 27.9%
        );
    }

    /**
     * Dell PowerEdge R720 - Medium performance balanced server
     * Idle: 98W, Peak: 345W (28.4% idle power)
     */
    public static HostProfile MEDIUM() {
        return new HostProfile(
            "Dell-R720-Medium",
            16,         // 16 cores
            50000,      // 50000 MIPS per core
            65536,      // 64 GB RAM
            20000,      // 20 Gbps
            200000,     // 200 GB storage
            345,        // 345W peak power
            28.4        // 98W idle / 345W = 28.4%
        );
    }

    /**
     * Cisco UCS C240 M4 - High performance server
     * Idle: 142W, Peak: 476W (29.8% idle power)
     */
    public static HostProfile HIGH_PERFORMANCE() {
        return new HostProfile(
            "Cisco-UCS-C240-HighPerf",
            24,         // 24 cores
            50000,      // 50000 MIPS per core
            131072,     // 128 GB RAM
            40000,      // 40 Gbps
            500000,     // 500 GB storage
            476,        // 476W peak power
            29.8        // 142W idle / 476W = 29.8%
        );
    }

    /**
     * HPE ProLiant DL380 Gen10 - Ultra high-end server
     * Idle: 194W, Peak: 634W (30.6% idle power)
     */
    public static HostProfile ULTRA_HIGH() {
        return new HostProfile(
            "HPE-DL380-Gen10-Ultra",
            32,         // 32 cores
            50000,      // 50000 MIPS per core
            262144,     // 256 GB RAM
            50000,      // 50 Gbps
            1000000,    // 1 TB storage
            634,        // 634W peak power
            30.6        // 194W idle / 634W = 30.6%
        );
    }

    // SPEC power_ssj2008 real server profiles (exact benchmark data)

    /**
     * Acer Altos R520 - Legacy small server (Intel Xeon E5450)
     * SPEC Data: Idle: 155W, Peak: 269W (57.6% idle power)
     * Very inefficient, represents older datacenter infrastructure
     */
    public static HostProfile SPEC_ACER_R520() {
        return new HostProfile(
            "Acer-Altos-R520",
            8,          // 8 cores (Xeon E5450)
            50000,      // 50000 MIPS per core
            16384,      // 16 GB RAM (typical for this generation)
            10000,      // 10 Gbps
            100000,     // 100 GB storage
            269,        // 269W peak power (SPEC)
            57.6        // 155W idle / 269W = 57.6%
        );
    }

    /**
     * Acer AR360 F2 - Medium older generation (Intel Xeon E5-2660)
     * SPEC Data: Idle: 69.4W, Peak: 315W (22.0% idle power)
     * Mid-range efficiency, represents transition-era servers
     */
    public static HostProfile SPEC_ACER_AR360() {
        return new HostProfile(
            "Acer-AR360-F2",
            16,         // 16 cores (Xeon E5-2660)
            50000,      // 50000 MIPS per core
            32768,      // 32 GB RAM
            10000,      // 10 Gbps
            200000,     // 200 GB storage
            315,        // 315W peak power (SPEC)
            22.0        // 69.4W idle / 315W = 22.0%
        );
    }

    /**
     * ASUSTeK RS720-E9/RS8 - Modern efficient server (Intel Xeon Platinum 8180)
     * SPEC Data: Idle: 48.2W, Peak: 385W (12.5% idle power)
     * Highly efficient modern server, excellent idle power characteristics
     */
    public static HostProfile SPEC_ASUS_RS720_E9() {
        return new HostProfile(
            "ASUSTeK-RS720-E9-RS8",
            56,         // 56 cores (Xeon Platinum 8180, 2x28)
            50000,      // 50000 MIPS per core
            131072,     // 128 GB RAM
            25000,      // 25 Gbps
            500000,     // 500 GB storage
            385,        // 385W peak power (SPEC)
            12.5        // 48.2W idle / 385W = 12.5%
        );
    }

    /**
     * ASUSTeK RS500A-E10-PS4 - Large AMD EPYC server (AMD EPYC 7742)
     * SPEC Data: Idle: 51.4W, Peak: 214W (24.0% idle power)
     * Extremely power-efficient, 64-core single socket
     */
    public static HostProfile SPEC_ASUS_RS500A() {
        return new HostProfile(
            "ASUSTeK-RS500A-E10-PS4",
            64,         // 64 cores (AMD EPYC 7742)
            50000,      // 50000 MIPS per core
            262144,     // 256 GB RAM
            25000,      // 25 Gbps
            1000000,    // 1 TB storage
            214,        // 214W peak power (SPEC)
            24.0        // 51.4W idle / 214W = 24.0%
        );
    }

    /**
     * ASUSTeK RS700A-E9-RS4V2 - Ultra large dual-socket (AMD EPYC 7742 x2)
     * SPEC Data: Idle: 106W, Peak: 430W (24.7% idle power)
     * High capacity server with 128 cores, good efficiency at scale
     */
    public static HostProfile SPEC_ASUS_RS700A() {
        return new HostProfile(
            "ASUSTeK-RS700A-E9-RS4V2",
            128,        // 128 cores (AMD EPYC 7742 x2)
            50000,      // 50000 MIPS per core
            524288,     // 512 GB RAM
            40000,      // 40 Gbps
            2000000,    // 2 TB storage
            430,        // 430W peak power (SPEC)
            24.7        // 106W idle / 430W = 24.7%
        );
    }

    public double getIdlePowerW() {
        return maxPowerW * staticPowerPercent / 100.0;
    }

    /**
     * Resolve profile name to HostProfile instance.
     *
     * @param name Profile name (case-insensitive)
     * @return HostProfile instance
     * @throws IllegalArgumentException if name is unknown
     */
    public static HostProfile fromName(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Profile name cannot be null or empty");
        }

        String upperName = name.toUpperCase().trim();

        switch (upperName) {
            case "LOW_POWER":
                return LOW_POWER();
            case "MEDIUM":
                return MEDIUM();
            case "HIGH_PERFORMANCE":
            case "HIGH_PERF":  // Alias
                return HIGH_PERFORMANCE();
            case "ULTRA_HIGH":
            case "ULTRA":  // Alias
                return ULTRA_HIGH();

            // SPEC profiles
            case "SPEC_ACER_R520":
            case "ACER_R520":  // Alias
                return SPEC_ACER_R520();
            case "SPEC_ACER_AR360":
            case "ACER_AR360":  // Alias
                return SPEC_ACER_AR360();
            case "SPEC_ASUS_RS720_E9":
            case "ASUS_RS720_E9":  // Alias
                return SPEC_ASUS_RS720_E9();
            case "SPEC_ASUS_RS500A":
            case "ASUS_RS500A":  // Alias
                return SPEC_ASUS_RS500A();
            case "SPEC_ASUS_RS700A":
            case "ASUS_RS700A":  // Alias
                return SPEC_ASUS_RS700A();

            default:
                throw new IllegalArgumentException(
                    "Unknown host profile: '" + name + "'. " +
                    "Available profiles: LOW_POWER, MEDIUM, HIGH_PERFORMANCE, ULTRA_HIGH, " +
                    "SPEC_ACER_R520, SPEC_ACER_AR360, SPEC_ASUS_RS720_E9, " +
                    "SPEC_ASUS_RS500A, SPEC_ASUS_RS700A"
                );
        }
    }

    @Override
    public String toString() {
        return String.format("%s [%d cores, %.0fW peak, %.0fW idle]",
            name, pes, maxPowerW, getIdlePowerW());
    }
}
