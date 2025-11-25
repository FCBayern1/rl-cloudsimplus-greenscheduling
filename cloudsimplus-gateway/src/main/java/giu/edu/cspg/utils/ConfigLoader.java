package giu.edu.cspg.utils;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;

public class ConfigLoader {
    private static final Logger LOGGER = LoggerFactory.getLogger(ConfigLoader.class.getSimpleName());
    private static final String DEFAULT_CONFIG_PATH = "config.yml"; // Default path relative to execution

    /**
     * Loads simulation parameters from a YAML configuration file.
     * 
     * @param configFilePath Path to the YAML configuration file. If null, uses
     *                       DEFAULT_CONFIG_PATH.
     * @param experimentId   The specific experiment ID (e.g., "experiment_1") to
     *                       load from the config file.
     * @return A Map containing the merged common and experiment-specific
     *         parameters.
     */
    public static Map<String, Object> loadConfig(String configFilePath, String experimentId) {
        String path = (configFilePath == null) ? DEFAULT_CONFIG_PATH : configFilePath;
        LOGGER.info("Loading configuration from: {} for experiment: {}", path, experimentId);
        Yaml yaml = new Yaml();
        try (InputStream inputStream = Files.newInputStream(Paths.get(path))) {
            Map<String, Object> fullConfig = yaml.load(inputStream);
            if (fullConfig == null) {
                LOGGER.error("Failed to load or parse YAML file: {}", path);
                return Collections.emptyMap();
            }

            Map<String, Object> commonParams = castToMap(fullConfig.getOrDefault("common", Collections.emptyMap()));
            Map<String, Object> experimentParams = castToMap(
                    fullConfig.getOrDefault(experimentId, Collections.emptyMap()));

            // Merge common and experiment-specific params, experiment overrides common
            Map<String, Object> mergedParams = new HashMap<>(commonParams);
            mergedParams.putAll(experimentParams); // Experiment-specific values overwrite common ones

            LOGGER.info("Successfully loaded and merged configuration for {}.", experimentId);
            return mergedParams;

        } catch (FileNotFoundException e) {
            LOGGER.error("Configuration file not found: {}", path, e);
            return Collections.emptyMap();
        } catch (Exception e) {
            LOGGER.error("Error loading or parsing configuration file: {}", path, e);
            return Collections.emptyMap();
        }
    }

    public static Map<String, Object> loadConfig(String experimentId) {
        return loadConfig(null, experimentId);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castToMap(Object obj) {
        if (!(obj instanceof Map)) {
            throw new IllegalArgumentException("Expected a Map but got: " + (obj == null ? "null" : obj.getClass()));
        }
        return (Map<String, Object>) obj;
    }
}
