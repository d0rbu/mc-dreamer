package org.d0rb;

import io.xol.enklume.MinecraftWorld;
import io.xol.enklume.MinecraftRegion;
import io.xol.enklume.MinecraftChunk;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Future;
import java.util.zip.DataFormatException;
import jdk.incubator.vector.IntVector;

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

    private final static int X_CHUNKS = 16;
    private final static int Z_CHUNKS = 16;
    private final static int X_BLOCKS = 16;
    private final static int Y_BLOCKS = 256;
    private final static int Z_BLOCKS = 16;
    private final static int NUM_BLOCKS = 256;
    private final static int NUM_VECS = NUM_BLOCKS / 8;
    private final static int VOLUME_SIDE = 16;

    private static void rollBlockCounts(IntVector[][][][] blockCounts) {
        for (int i = 0; i < VOLUME_SIDE; i++) {
            blockCounts[i] = blockCounts[i + 1];
        }
        
        for (int y = 0; y < Y_BLOCKS; y++) {
            for (int z = 0; z < Z_CHUNKS * Z_BLOCKS; z++) {
                for (int i = 0; i < NUM_VECS; i++) {
                    blockCounts[VOLUME_SIDE][y][z][i] = IntVector.zero(IntVector.SPECIES_256);
                }
            }
        }
    }

    private static void processRegionFile(File regionFile, String worldName) throws DataFormatException, IOException {
        String regionFileName = regionFile.getName();
        String regionFileRoot = regionFileName.substring(0, regionFileName.length() - 4);
        String regionRatingName = worldName + "/" + regionFileRoot + ".regionrating";

        // For an element at (i, j, k), this is the rating for the volume from (i - 15, j - 15, k - 15) to (i + 1, j + 1, k + 1).
        float[][][] volumeRating = new float[X_CHUNKS * X_BLOCKS][Y_BLOCKS][Z_CHUNKS * Z_BLOCKS];
        // Block counts for the YZ plane we are on and the one preceding us.
        IntVector[][][][] blockCounts = new IntVector[VOLUME_SIDE + 1][Y_BLOCKS][Z_CHUNKS * Z_BLOCKS][NUM_VECS];
        for (int i = 0; i < VOLUME_SIDE + 1; i++) {
            for (int j = 0; j < Y_BLOCKS; j++) {
                for (int k = 0; k < Z_CHUNKS * Z_BLOCKS; k++) {
                    for (int l = 0; l < NUM_VECS; l++) {
                        blockCounts[i][j][k][l] = IntVector.zero(IntVector.SPECIES_256);
                    }
                }
            }
        }

        MinecraftRegion region = new MinecraftRegion(regionFile);

        for (int chunkX = 0; chunkX < X_CHUNKS; chunkX++) {
            for (int chunkZ = 0; chunkZ < Z_CHUNKS; chunkZ++) {
                MinecraftChunk chunk = region.getChunk(chunkX, chunkZ);
                for (int blockX = 0; blockX < X_BLOCKS; blockX++) {
                    rollBlockCounts(blockCounts);
                    for (int y = 0; y < Y_BLOCKS; y++) {
                        for (int blockZ = 0; blockZ < Z_BLOCKS; blockZ++) {
                            int x = chunkX * X_BLOCKS + blockX;
                            int z = chunkZ * Z_BLOCKS + blockZ;

                            int blockId = chunk.getBlockID(blockX, y, blockZ);

                            if (y > 0) {
                                if (z > 0) {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        blockCounts[VOLUME_SIDE][y][z][i] =
                                                blockCounts[VOLUME_SIDE-1][y][z][i].add(
                                                blockCounts[VOLUME_SIDE][y - 1][z][i]).add(
                                                blockCounts[VOLUME_SIDE][y][z - 1][i]).sub(
                                                blockCounts[VOLUME_SIDE][y - 1][z - 1][i]).sub(
                                                blockCounts[VOLUME_SIDE-1][y - 1][z][i]).sub(
                                                blockCounts[VOLUME_SIDE-1][y][z - 1][i]).add(
                                                blockCounts[VOLUME_SIDE-1][y - 1][z - 1][i]);
                                    }
                                } else {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        blockCounts[VOLUME_SIDE][y][z][i] =
                                                blockCounts[VOLUME_SIDE-1][y][z][i].add(
                                                blockCounts[VOLUME_SIDE][y - 1][z][i]).sub(
                                                blockCounts[VOLUME_SIDE-1][y - 1][z][i]);
                                    }
                                }
                            } else {
                                if (z > 0) {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        blockCounts[VOLUME_SIDE][y][z][i] =
                                                blockCounts[VOLUME_SIDE-1][y][z][i].add(
                                                blockCounts[VOLUME_SIDE][y][z - 1][i]).sub(
                                                blockCounts[VOLUME_SIDE-1][y][z - 1][i]);
                                    }
                                } else {
                                    System.arraycopy(blockCounts[VOLUME_SIDE - 1][y][z], 0, blockCounts[VOLUME_SIDE][y][z], 0, NUM_VECS);
                                }
                            }

                            int vectorIdx = blockId / 8;
                            int bitIdx = blockId % 8;
                            int[] countVector = new int[8];
                            countVector[bitIdx] = 1;
                            blockCounts[VOLUME_SIDE][y][z][vectorIdx] = blockCounts[VOLUME_SIDE][y][z][vectorIdx].add(IntVector.fromArray(IntVector.SPECIES_256, countVector, 0));

                            IntVector[] volumeBlockCounts = new IntVector[NUM_VECS];
                            if (y >= VOLUME_SIDE) {
                                if (z >= VOLUME_SIDE) {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        volumeBlockCounts[i] = 
                                                blockCounts[VOLUME_SIDE][y][z][i].sub(
                                                blockCounts[0][y][z][i]).sub(
                                                blockCounts[VOLUME_SIDE][y - VOLUME_SIDE][z][i]).sub(
                                                blockCounts[VOLUME_SIDE][y][z - VOLUME_SIDE][i]).add(
                                                blockCounts[VOLUME_SIDE][y - VOLUME_SIDE][z - VOLUME_SIDE][i]).add(
                                                blockCounts[0][y - VOLUME_SIDE][z][i]).add(
                                                blockCounts[0][y][z - VOLUME_SIDE][i]).sub(
                                                blockCounts[0][y - VOLUME_SIDE][z - VOLUME_SIDE][i]);
                                    }
                                } else {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        volumeBlockCounts[i] = 
                                                blockCounts[VOLUME_SIDE][y][z][i].sub(
                                                blockCounts[0][y][z][i]).sub(
                                                blockCounts[VOLUME_SIDE][y - VOLUME_SIDE][z][i]).add(
                                                blockCounts[0][y - VOLUME_SIDE][z][i]);
                                    }
                                }
                            } else {
                                if (z >= VOLUME_SIDE) {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        volumeBlockCounts[i] = 
                                                blockCounts[VOLUME_SIDE][y][z][i].sub(
                                                blockCounts[0][y][z][i]).sub(
                                                blockCounts[VOLUME_SIDE][y][z - VOLUME_SIDE][i]).add(
                                                blockCounts[0][y][z - VOLUME_SIDE][i]);
                                    }
                                } else {
                                    for (int i = 0; i < NUM_VECS; i++) {
                                        volumeBlockCounts[i] = 
                                                blockCounts[VOLUME_SIDE][y][z][i].sub(
                                                blockCounts[0][y][z][i]);
                                    }
                                }
                            }

                            // TODO: use block counts to compute rating
                        }
                    }
                }
            }
        }
    }
}