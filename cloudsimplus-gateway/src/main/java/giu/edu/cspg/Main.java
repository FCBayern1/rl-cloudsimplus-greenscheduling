package giu.edu.cspg;

import java.net.InetAddress;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.singledc.LoadBalancerGateway;
import py4j.CallbackClient;
import py4j.GatewayServer;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class.getSimpleName());

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
        LoadBalancerGateway simulationEnvironment = new LoadBalancerGateway();
        InetAddress all = InetAddress.getByName("0.0.0.0");
        int gatewayPort = resolveGatewayPort(args);
        GatewayServer gatewayServer = new GatewayServer(
                simulationEnvironment,
                gatewayPort,
                all,
                GatewayServer.DEFAULT_CONNECT_TIMEOUT,
                GatewayServer.DEFAULT_READ_TIMEOUT,
                null,
                new CallbackClient(GatewayServer.DEFAULT_PYTHON_PORT, all));
        logger.info("Starting server: " + gatewayServer.getAddress() + " " + gatewayServer.getPort());
        gatewayServer.start();
        simulationEnvironment.setGatewayServer(gatewayServer);
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
