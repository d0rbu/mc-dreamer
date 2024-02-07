package org.d0rb;

import io.xol.enklume.MinecraftWorld;
import io.xol.enklume.MinecraftRegion;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Future;
import java.util.zip.DataFormatException;

public class Main {
    public static void main(String[] args) throws java.io.IOException, DataFormatException {
        if (args.length != 1) {
            System.out.println("Pass the path to the world folder as the only argument.");
            return;
        }

        String worldPath = args[0];
        File worldFolder = new File(worldPath);
        String worldName = worldFolder.getName();

        MinecraftWorld world = new MinecraftWorld(worldFolder);

        List<File> regionFiles = world.getAllRegionFiles(0);

        // Spawn a thread for each region file to process them in parallel
        for (File regionFile : regionFiles) {
            processRegionFile(regionFile, worldName);
            break;
        }
    }

    private static Future<Void> processRegionFile(File regionFile, String worldName) throws DataFormatException, IOException {
        String regionFileName = regionFile.getName();
        String regionFileRoot = regionFileName.substring(0, regionFileName.length() - 4);
        String regionRatingName = worldName + "/" + regionFileRoot + ".regionrating";

        float[][][] regionRating = new float[16*32][256][16*32];
        // int[][][][] regionIntegral = new int[16*32][256][16*32][256];  // 256 is just a placeholder for now, we'll figure out the actual number of dimensions later

        MinecraftRegion region = new MinecraftRegion(regionFile);

        return null;
    }
}