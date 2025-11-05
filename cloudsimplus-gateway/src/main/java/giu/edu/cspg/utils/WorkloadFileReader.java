package giu.edu.cspg.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import static org.apache.commons.lang3.StringUtils.isNumeric;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.traces.TraceReaderAbstract;

import giu.edu.cspg.common.CloudletDescriptor;

/**
 * Reads workload traces from SWF or CSV files and generates a list
 * of {@link CloudletDescriptor} objects.
 */
public final class WorkloadFileReader extends TraceReaderAbstract {
    // --- Constants for SWF Parsing (based on SwfWorkloadFileReader) ---
    public static final String SWF_DELIMITER = "\\s+";
    private static final int SWF_JOB_NUM_INDEX = 0;
    private static final int SWF_SUBMIT_TIME_INDEX = 1;
    private static final int SWF_RUN_TIME_INDEX = 3;
    private static final int SWF_NUM_PROC_INDEX = 4; // Actual PEs used
    private static final int SWF_REQ_NUM_PROC_INDEX = 7; // Requested PEs
    private static final int SWF_STATUS_INDEX = 10; // Job status (0=Failed)
    private static final int SWF_FIELD_COUNT = 18; // Minimum expected fields
    private static final int SWF_IRRELEVANT = -1; // Placeholder for irrelevant fields

    // --- Constants for CSV Parsing ---
    // Header: job_id,arrival_time,mi,allocated_cores
    private static final String CSV_DELIMITER = ",";
    private static final int CSV_ID_INDEX = 0;
    private static final int CSV_SUBMIT_TIME_INDEX = 1;
    private static final int CSV_MI_INDEX = 2;
    private static final int CSV_CORES_INDEX = 3;
    private static final int CSV_FIELD_COUNT = 4;

    private final String workloadMode; // "SWF" or "CSV"
    private int referenceMips; // Only used for SWF
    private final List<CloudletDescriptor> descriptors; // List of Cloudlet descriptors generated from the trace file

    /**
     * A {@link Predicate} which indicates when a {@link CloudletDescriptor}
     * must be created from a trace line read from the workload file.
     * If a Predicate is not set, a CloudletDescriptor will be created for any line
     * read.
     */
    private Predicate<CloudletDescriptor> predicate;

    /**
     * Creates a WorkloadFileReader.
     *
     * @param filePath      Path to the trace file (.swf, .gz, .zip, .csv).
     * @param workloadMode  "SWF" or "CSV".
     * @param referenceMips MIPS value used to calculate Cloudlet MI from SWF
     *                      runtime. Ignored for CSV.
     */
    public WorkloadFileReader(String filePath, String workloadMode, int referenceMips) {
        super(filePath);
        if (!"SWF".equalsIgnoreCase(workloadMode) && !"CSV".equalsIgnoreCase(workloadMode)) {
            throw new IllegalArgumentException("Invalid workloadMode. Must be 'SWF' or 'CSV'.");
        }
        if ("SWF".equalsIgnoreCase(workloadMode) && referenceMips <= 0) {
            throw new IllegalArgumentException("Reference MIPS must be positive for SWF mode.");
        }
        this.workloadMode = workloadMode;
        this.setReferenceMips(referenceMips);
        this.descriptors = new ArrayList<>();

        /*
         * A default predicate which indicates that a CloudletDescriptor will be
         * created for any job read from the workload reader.
         * That is, there isn't an actual condition to create a CloudletDescriptor.
         */
        this.predicate = cloudletDescriptor -> true;
    }

    /**
     * Reads the trace file based on the configured mode and generates descriptors.
     * 
     * @return A list of CloudletDescriptor objects.
     */
    public List<CloudletDescriptor> generateDescriptors() {
        if (descriptors.isEmpty()) {
            if ("SWF".equalsIgnoreCase(workloadMode)) {
                this.setFieldDelimiterRegex(SWF_DELIMITER);
                readFile(this::createCloudletDescriptorFromSwfLine);
            } else {
                this.setFieldDelimiterRegex(CSV_DELIMITER);
                readFile(this::createCloudletDescriptorFromCsvLine);
            }
        }
        return descriptors;
    }

    private Boolean createCloudletDescriptorFromSwfLine(final String[] parsedLineArray) {
        // If all the fields couldn't be read, don't create the Cloudlet Descriptor.
        if (parsedLineArray.length < SWF_FIELD_COUNT) {
            return false;
        }

        // Check the job status field to filter out jobs that are not completed.
        if (Integer.parseInt(parsedLineArray[SWF_STATUS_INDEX].trim()) == 0) {
            return false;
        }

        int parsedJobId = Integer.parseInt(parsedLineArray[SWF_JOB_NUM_INDEX].trim());

        final int id = parsedJobId <= SWF_IRRELEVANT ? descriptors.size() + 1
                : parsedJobId;

        /*
         * according to the SWF manual, runtime of 0 is possible due
         * to rounding down. E.g. runtime is 0.4 seconds -> runtime = 0
         */
        final int runTime = Math.max(Integer.parseInt(parsedLineArray[SWF_RUN_TIME_INDEX].trim()), 1);

        /*
         * if the required num of allocated processors field is ignored
         * or zero, then use the actual field
         */
        final int maxNumProc = Math.max(
                Integer.parseInt(parsedLineArray[SWF_REQ_NUM_PROC_INDEX].trim()),
                Integer.parseInt(parsedLineArray[SWF_NUM_PROC_INDEX].trim()));
        final int numProc = Math.max(1, maxNumProc);

        long submitTime = Long.parseLong(parsedLineArray[SWF_SUBMIT_TIME_INDEX].trim());
        submitTime = Math.max(0, submitTime);

        // Calculate MI based on runtime and reference MIPS
        // (runTime * MIPS) = Instructions processed by one PE in that time.
        int mi = runTime * this.referenceMips;
        mi = Math.max(1, mi);

        final CloudletDescriptor descriptor = new CloudletDescriptor(id, submitTime, mi, numProc);

        if (predicate.test(descriptor)) {
            descriptors.add(descriptor);
            return true;
        }

        return false;
    }

    private Boolean createCloudletDescriptorFromCsvLine(final String[] parsedLineArray) {
        // If all the fields couldn't be read, don't create the Cloudlet Descriptor.
        if (parsedLineArray.length < CSV_FIELD_COUNT) {
            return false;
        }

        // Skip the header line if present
        if (!isNumeric(parsedLineArray[CSV_ID_INDEX].trim())) {
            return false;
        }

        int cloudletId = Integer.parseInt(parsedLineArray[CSV_ID_INDEX].trim());
        int submissionDelay = Integer.parseInt(parsedLineArray[CSV_SUBMIT_TIME_INDEX].trim());
        int mi = Integer.parseInt(parsedLineArray[CSV_MI_INDEX].trim());
        int numberOfCores = Integer.parseInt(parsedLineArray[CSV_CORES_INDEX].trim());

        // Basic validation ensuring minimum values for each field
        submissionDelay = Math.max(0, submissionDelay);
        mi = Math.max(1, mi);
        numberOfCores = Math.max(1, numberOfCores);

        final CloudletDescriptor descriptor = new CloudletDescriptor(cloudletId, submissionDelay, mi, numberOfCores);

        if (predicate.test(descriptor)) {
            descriptors.add(descriptor);
            return true;
        }

        return false;
    }

    /**
     * Sets the MIPS capacity of the PEs from the VM where each created Cloudlet is
     * supposed to run.
     * Considering the workload reader provides the run time for each
     * application registered inside the reader, the MIPS value will be used
     * to compute the {@link Cloudlet#getLength() length of the Cloudlet (in MI)}
     * so that it's expected to execute, inside the VM with the given MIPS capacity,
     * for the same time as specified into the workload reader.
     *
     * @param referenceMips the MIPS value to set
     */
    public WorkloadFileReader setReferenceMips(final int referenceMips) {
        if (referenceMips <= 0) {
            throw new IllegalArgumentException("MIPS must be greater than 0.");
        }
        this.referenceMips = referenceMips;
        return this;
    }
}
