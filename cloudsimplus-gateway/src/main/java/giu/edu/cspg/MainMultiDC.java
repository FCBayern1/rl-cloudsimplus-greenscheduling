package giu.edu.cspg;

import java.net.InetAddress;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.multidc.HierarchicalMultiDCGateway;
import py4j.CallbackClient;
import py4j.GatewayServer;

/**
 * Main class for running the Hierarchical Multi-Datacenter Gateway.
 * This supports the hierarchical MARL environment with global and local agents.
 */
public class MainMultiDC {

    private static final Logger logger = LoggerFactory.getLogger(MainMultiDC.class.getSimpleName());

    public static void main(String[] args) throws Exception {
        logger.info("Starting Hierarchical Multi-Datacenter Gateway...");

        // Create the hierarchical multi-DC gateway instance
        HierarchicalMultiDCGateway multiDCGateway = HierarchicalMultiDCGateway.getInstance();

        // Configure Py4J gateway server
        InetAddress all = InetAddress.getByName("0.0.0.0");
        GatewayServer gatewayServer = new GatewayServer(
                multiDCGateway,
                GatewayServer.DEFAULT_PORT,
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