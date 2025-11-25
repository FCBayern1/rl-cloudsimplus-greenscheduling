package giu.edu.cspg.tables;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;

import org.cloudsimplus.builders.tables.AbstractTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TableLogger {

    private static final Logger LOGGER = LoggerFactory.getLogger(TableLogger.class.getSimpleName());

    public static void logAndSaveTable(AbstractTable table, String filePath) {
        // Step 1: Create output stream to capture printed content
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream customPrintStream = new PrintStream(baos);

        // Step 2: Redirect table output to custom stream
        table.setPrintStream(customPrintStream);

        // Step 3: Print table (captured in baos)
        table.print();

        // Step 4: Convert output to String
        String output = baos.toString();

        // Step 5: Remove trailing newline (if any)
        String trimmedOutput = output.stripTrailing();

        // Step 6: Log the output
        LOGGER.info("\n{}", trimmedOutput);

        // Step 7: Write to file
        try {
            File file = new File(filePath);
            file.getParentFile().mkdirs(); // ensure directory exists
            Files.writeString(file.toPath(), trimmedOutput);
        } catch (IOException e) {
            LOGGER.error("Failed to write CSV to {}: {}", filePath, e.getMessage(), e);
        }
    }
}
