package Java.Algorithms;

import java.util.HashSet;
import java.util.Set;

public class Sets {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common set algorithms ---");

        itemExistsInSet();

        return;
    }

    /*
     * Sets can be used to check if a unique element exists with Time Complexity O(1)
     * Example Provided - Checking if a set of numbers contains item k
     */
    private static void itemExistsInSet() {
        int[] numbers = {1, 2, 3, 4, 5};
        Set<Integer> numberSet = new HashSet<>();
        for (int num : numbers) {
            numberSet.add(num);
        }

        long start = System.nanoTime();
        boolean containsInt = numberSet.contains(3);
        long end = System.nanoTime();

        System.out.println("Found number in set: " + containsInt + " - Time to Execute: " + (end - start));
    }
}
