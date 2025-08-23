package Java.Algorithms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Maps {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common maps algorithms ---");

        frequencyMap();
        twoSumMethod();

        return;
    }

    /*
     * This method demonstrates two easy ways to do operations on a map.
     * A maps insert, get, and contains methods run in O(1) time complexity on average
     */
    private static void frequencyMap() {
        List<String> stringsList = List.of("One", "Two", "Three", "Four", "Five");
        Map<String, Integer> stringsMap = new HashMap<>();

        for (String item : stringsList) {
            if (!stringsMap.containsKey(item)) {
                stringsMap.put(item, 1);
            }
            else {
                stringsMap.put(item, stringsMap.get(item) + 1);
            }
        }
        System.out.println("Printing Map: " + stringsMap.toString());

        // Alternative if-else statement
        for (String item : stringsList) {
            stringsMap.put(item, stringsMap.getOrDefault(item, 1) + 1);
        }
        System.out.println("Printing Map: " + stringsMap.toString());

        return;
    }

    /*
     * Map Implementation where the complement of a number given a target is stored
     * O(n) Time Complexity
     * Example Provided: 
     */
    private static void twoSumMethod() {
        int[] nums = {2, 7, 11, 15};
        int target = 9;
        Map<Integer, Integer> intMap = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (intMap.containsKey(complement)) {
                System.out.println("First Index: " + intMap.get(complement) + " Second Index: " + i);
                return;
            }
            else {
                intMap.put(nums[i], i);
            }
        }

        return;
    }
}
