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

    public static void main(String[] args) throws Exception {
        LoadBalancerGateway simulationEnvironment = new LoadBalancerGateway();
        InetAddress all = InetAddress.getByName("0.0.0.0");
        GatewayServer gatewayServer = new GatewayServer(
                simulationEnvironment,
                GatewayServer.DEFAULT_PORT,
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
