package Java;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/*
 * Common Time and Space Complexities 
 * 
 * O(1) Constant - Set Lookups, Accessing an Array Index
 * O(log n) Logarithmic - Binary Search, Splitting data in half each time
 * O(n) - Linear - Each item once, loops & traversing lists
 * O(n log n) - Sorting, default sort methods
 * O(n^2) - Nested loops for->for, brute force comparisons
 * O(2^n)
 * O(n!)
 * 
 * For inputs 10^5, need O(n log n) or better
 * For inputs 10^4, need O(n^2) or better
 * 
 */

public class Algorithms {
    public static void main(String[] args) {
        System.out.println("Entry Point");

        arrays();
        strings();
        sets();
        maps();
    }

    private static void arrays() {
        // Two Pointer implementation where each pointer starts at the ends of the array
        // O(n) 
        // Case insensitive, only alphanumeric characters, ignore spaces
        String palindrome = "A man, a plan, a canal: Panama";
        int left = 0;
        int right = palindrome.length() - 1;

        while (left < right) {
            while ((left < right) && (!Character.isLetterOrDigit(palindrome.charAt(left)))) {
                left = left + 1;
            }
            while ((left < right) && (!Character.isLetterOrDigit(palindrome.charAt(right)))) {
                right = right - 1;
            }

            if (Character.toLowerCase(palindrome.charAt(left)) != Character.toLowerCase(palindrome.charAt(right))) {
                System.out.println("Not a Valid Palindrome");
            }

            left = left + 1;
            right = right - 1;
        }
        System.out.println("Valid Palindrome");
    }

    private static void strings() {
        String original = "input";

        // Tracking time
        long startTime;
        long endTime;

        // Bad Practice, creates a new string every time a character is appended
        // instead of modifying the existing string because strings are immutable. O(n^2)
        startTime = System.nanoTime();
        String result = "";
        for (char character : original.toCharArray()) {
            result = result + character;
        }
        endTime = System.nanoTime();
        System.out.println("String Concatenation Result: " + result + " Time to Execute: " + (endTime - startTime));

        // This has a better time complexity O(n) because it is faster to build
        // an array of characters and convert into a string instead of creating a new 
        // string every time you append a character
        startTime = System.nanoTime();
        char[] charResult = new char[original.length()];
        int i = 0;
        for (char character : original.toCharArray()) {
            charResult[i] = character;
            i++;
        }
        String resultString = new String(charResult);
        endTime = System.nanoTime();
        System.out.println("Character Array Result: " + resultString + " Time to Execute: " + (endTime - startTime));

        // More common Java approach
        startTime = System.nanoTime();
        StringBuilder stringBuilder = new StringBuilder();
        for (char character : original.toCharArray()) {
            stringBuilder.append(character);
        }
        String builderResult = stringBuilder.toString();
        endTime = System.nanoTime();
        System.out.println("String Builder Result: " + builderResult + " Time to Execute: " + (endTime - startTime));
    }

    private static void sets() {
        int[] numbers = {1, 2, 3, 4, 5};

        // Tracking time
        long startTime;
        long endTime;

        // Checking if an item exists in a set should be O(1) time
        Set<Integer> numbersSet = new HashSet<>();
        for (int num : numbers) {
            numbersSet.add(num);
        }
        startTime = System.nanoTime();
        boolean exists = numbersSet.contains(3);
        endTime = System.nanoTime();

        System.out.println("Checking if item exists: " + exists + " Time to Execute: " + (endTime - startTime));
    }

    private static void maps() {
        List<String> stringsList = List.of("One", "Two", "One", "Three");
        Map<String, Integer> myMap = new HashMap<>();

        // Frequency Map
        for (String item : stringsList) {
            if (!myMap.containsKey(item)) {
                myMap.put(item, 1);
            }
            else {
                myMap.put(item, myMap.get(item) + 1);
            }
        }
        System.out.println("Printing Map: " + myMap.toString());

        Map<String, Integer> myOtherMap = new HashMap<>();

        // Alternative
        for (String item : stringsList) {
            myOtherMap.put(item, myOtherMap.getOrDefault(item, 0) + 1);
        }
        System.out.println("Printing Other Map: " + myOtherMap.toString());

        // Two Sum example
        // O(n) Time Complexity and Space Complexity
        int[] nums = {2, 7, 11, 15};
        int target = 9;
        Map<Integer, Integer> integerMap = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (integerMap.containsKey(complement)) {
                System.out.println("First Index: " + integerMap.get(complement) + " Second Index: " + i);
            }
            else {
                integerMap.put(nums[i], i);
            }
        }
    }
}