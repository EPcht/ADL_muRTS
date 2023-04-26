import java.net.ServerSocket;
import java.net.Socket;

import java.lang.reflect.Constructor;

import rts.Game;
import rts.GameSettings;
import rts.GameSettings.LaunchMode;
import rts.units.UnitTypeTable;

import ai.RandomAI;
import ai.core.AI;
import ai.socket.SocketAI;

public class ADL {

    public static void main(String[] args) throws Exception
    {        
        if (args.length != 1)
            return;

        try {
            System.out.println("Trying to create a server on port : " + args[0] + "...");
            ServerSocket serverSocket = new ServerSocket(Integer.parseInt(args[0]));
            Socket socket = serverSocket.accept();
                    
            // Argument à récupérer via ligne de commande
            String serverAddress = "localhost";
            int serverPort = Integer.parseInt(args[0]);

            // Argument à récupérer via envoie de message
            int serializationType = 2; // JSON
            String mapLocation = "../maps/8x8/bases8x8.xml";
            int maxCycles = 5000;
            int updateInterval = 20;
            boolean partiallyObservable = false;
            int uttVersion = 2;
            // NSP
            int conflictPolicy = 1;
            LaunchMode launchMode = LaunchMode.valueOf("STANDALONE");
            // NSP
            boolean includeConstantsInState = true;
            // NSP
            boolean compressTerrain = false;
            boolean headless = false;
            
            // Argument à récupérer via envoie de message à chaque tour de boucle
            String AI1 = "ai.RandomAI";
            String AI2 = "ai.RandomAI";

            GameSettings gameSettings = new GameSettings(launchMode, serverAddress, serverPort, serializationType, mapLocation, maxCycles, updateInterval, partiallyObservable, uttVersion, conflictPolicy, includeConstantsInState, compressTerrain, headless, AI1, AI2);

            UnitTypeTable unitTypeTable = new UnitTypeTable(
                gameSettings.getUTTVersion(), gameSettings.getConflictPolicy());

            // Generate players
            // player 1 is created from SocketAI
            AI player_one = SocketAI.createFromExistingSocket(100, 0, unitTypeTable,
                gameSettings.getSerializationType(), gameSettings.isIncludeConstantsInState(),
                gameSettings.isCompressTerrain(), socket);
            // player 2 is created using the info from gameSettings
            Constructor cons2 = Class.forName(gameSettings.getAI2())
                .getConstructor(UnitTypeTable.class);
            AI player_two = (AI) cons2.newInstance(unitTypeTable);

            Game game = new Game(gameSettings, player_one, player_two);
            System.out.println("Starting the game...");
            game.start();

        } catch (Exception e ) {
            e.printStackTrace();
        }
    }
}

/*            
System.out.println("Trying to create a server on port : " + args[0] + "...");
ServerSocket serverSocket = new ServerSocket(Integer.parseInt(args[0]));
Socket socket = serverSocket.accept();
        
// Argument à récupérer via ligne de commande
String serverAddress = "localhost";
int serverPort = Integer.parseInt(args[0]);

// Argument à récupérer via envoie de message
int serializationType = 2; // JSON
String mapLocation = "../maps/8x8/bases8x8.xml";
int maxCycles = 5000;
int updateInterval = 20;
boolean partiallyObservable = true;
int uttVersion = 2;
// NSP
int conflictPolicy = 1;
LaunchMode launchMode = LaunchMode.valueOf("STANDALONE");
// NSP
boolean includeConstantsInState = true;
// NSP
boolean compressTerrain = false;
boolean headless = false;

int serializationType = readIntegerProperty(prop, "serialization_type", 2);
String mapLocation = prop.getProperty("map_location");
int maxCycles = readIntegerProperty(prop, "max_cycles", 5000);
int updateInterval = readIntegerProperty(prop, "update_interval", 20);
boolean partiallyObservable = Boolean.parseBoolean(prop.getProperty("partially_observable"));
int uttVersion = readIntegerProperty(prop, "UTT_version", 2);
// NSP
int conflictPolicy = readIntegerProperty(prop, "conflict_policy", 1);
LaunchMode launchMode = LaunchMode.valueOf(prop.getProperty("launch_mode",  "GUI"));
// NSP
boolean includeConstantsInState = Boolean.parseBoolean(prop.getProperty("constants_in_state", "true"));
// NSP
boolean compressTerrain = Boolean.parseBoolean(prop.getProperty("compress_terrain", "false"));
boolean headless = Boolean.parseBoolean(prop.getProperty("headless", "false"));

// Argument à récupérer via envoie de message à chaque tour de boucle
String AI1 = prop.getProperty("AI1", "ai.RandomAI");
String AI2 = prop.getProperty("AI2", "ai.RandomAI");
*/