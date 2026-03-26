package exe.edu.cspg;

import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import exe.edu.cspg.multidc.HierarchicalMultiDCGateway;
import py4j.CallbackClient;
import py4j.GatewayServer;

/**
 * Main class for running the Hierarchical Multi-Datacenter Gateway.
 * This supports the hierarchical MARL environment with global and local agents.
 */
public class MainMultiDC {

    private static final Logger logger = LoggerFactory.getLogger(MainMultiDC.class.getSimpleName());

    // #region agent log
    private static void writeDebugLog(String hypothesisId, String location, String message, String dataJson) {
        try {
            String payload = String.format(
                    "{\"sessionId\":\"f7b29b\",\"runId\":\"pre-fix\",\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\",\"data\":%s,\"timestamp\":%d}%n",
                    hypothesisId,
                    location,
                    message,
                    dataJson,
                    System.currentTimeMillis());
            Files.writeString(
                    Path.of("/home/joshua/rl-cloudsimplus-greenscheduling/.cursor/debug-f7b29b.log"),
                    payload,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (Exception ignored) {
        }
    }
    // #endregion

    private static int resolveGatewayPort(String[] args) {
        // Priority:
        // 1) CLI: --port <n>  (or --py4j-port <n>)
        // 2) CLI: first positional arg as port (backward compatible)
        // 3) Env: PY4J_PORT or CSPG_PY4J_PORT
        // 4) Default: GatewayServer.DEFAULT_PORT (25333)
        int port = GatewayServer.DEFAULT_PORT;

        if (args != null) {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if ("--port".equals(a) || "--py4j-port".equals(a) || "--py4j_port".equals(a)) {
                    if (i + 1 < args.length) {
                        try {
                            return Integer.parseInt(args[i + 1]);
                        } catch (NumberFormatException ignored) {
                            // fall through to other methods
                        }
                    }
                }
            }
            if (args.length >= 1) {
                try {
                    return Integer.parseInt(args[0]);
                } catch (NumberFormatException ignored) {
                    // ignore
                }
            }
        }

        String env = System.getenv("CSPG_PY4J_PORT");
        if (env == null || env.isBlank()) {
            env = System.getenv("PY4J_PORT");
        }
        if (env != null && !env.isBlank()) {
            try {
                port = Integer.parseInt(env.trim());
            } catch (NumberFormatException ignored) {
                // ignore and use default
            }
        }

        return port;
    }

    public static void main(String[] args) throws Exception {
        logger.info("Starting Hierarchical Multi-Datacenter Gateway...");

        // Create the hierarchical multi-DC gateway instance
        HierarchicalMultiDCGateway multiDCGateway = HierarchicalMultiDCGateway.getInstance();

        // Configure Py4J gateway server
        InetAddress all = InetAddress.getByName("0.0.0.0");
        int gatewayPort = resolveGatewayPort(args);
        // Use the standard CallbackClient object expected by Py4J's constructor.
        // This matches the single-DC gateway startup path and avoids the NPE seen
        // when cbClient is null during multi-worker JVM startup.
        // #region agent log
        writeDebugLog(
                "H1",
                "MainMultiDC.java:97",
                "construct_gateway_server",
                String.format("{\"gatewayPort\":%d,\"callbackClientMode\":\"default_python_port\",\"argsCount\":%d}", gatewayPort, args == null ? 0 : args.length));
        // #endregion
        GatewayServer gatewayServer = new GatewayServer(
                multiDCGateway,
                gatewayPort,
                all,
                GatewayServer.DEFAULT_CONNECT_TIMEOUT,
                GatewayServer.DEFAULT_READ_TIMEOUT,
                null,
                new CallbackClient(GatewayServer.DEFAULT_PYTHON_PORT, all));

        logger.info("Starting Py4J server: " + gatewayServer.getAddress() + " " + gatewayServer.getPort());
        gatewayServer.start();

        logger.info("HierarchicalMultiDCGateway is ready. Waiting for Python connections...");
    }

    public static void initiateShutdown(final GatewayServer gatewayServer) {
        try {
            Thread.sleep(2000); // wait for 2 seconds
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Interrupted", e);
        }
        Executors.newSingleThreadExecutor().execute(() -> {
            try {
                // Shutdown the Py4J gateway
                gatewayServer.shutdown();
                logger.info("Gateway server shut down.");

                // Terminate the JVM
                System.exit(0);
            } catch (Exception e) {
                logger.error("Error during shutdown", e);
            }
        });
    }
}