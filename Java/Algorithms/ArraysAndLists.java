package Java.Algorithms;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ArraysAndLists {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common array and list algorithms ---");

        oppositeEndTwoPointer();
        slowFastTwoPointer();
        fixedSlidingWindow();
        dynamicSlidingWindowWithSet();
        dynamicSlidingWindowWithMap();
        binarySearchWithNumbers();
        binarySearchWithCondition();

        return;
    }

    /*
     * Two Pointer Implementation where two pointers are initialized at opposite ends of an array
     * O(n)
     * Example Provided: Valid, Case Insensitive Palindrome Ignoring Non-Alphanumeric Characters
     */
    private static void oppositeEndTwoPointer() {
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
                System.out.println("Not a valid palindrome");
                return;
            }

            left = left + 1;
            right = right - 1;
        }

        System.out.println("Valid Palindrome");
        return;
    }

    /*
     * Simple Linked List Node Implementation
     * Singly Linked List, meaning nodes only point to the next node, and not the previous
     */
    static class ListNode {
        int val;
        ListNode next;

        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    /*
     * Two Pointer Implementation using slow/fast pointers where one moves one step and the other moves two steps.
     * O(n)
     * Contains both odd and even implementations
     * Example Provided: Find the middle value of a linked list
     */
    private static void slowFastTwoPointer() {
        // Odd Implementation, should land on 2
        ListNode linkedListOne = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4)))));
        ListNode slowOne = linkedListOne;
        ListNode fastOne = linkedListOne;

        while (fastOne != null && fastOne.next != null) {
            fastOne = fastOne.next.next;
            slowOne = slowOne.next;
        }
        System.out.println("Middle value of odd size linked list: " + slowOne.val);

        // Even Implementation, should land in 3
        ListNode linkedListTwo = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5))))));
        ListNode slowTwo = linkedListTwo;
        ListNode fastTwo = linkedListTwo;

        while (fastTwo != null && fastTwo.next != null) {
            fastTwo = fastTwo.next.next;
            slowTwo = slowTwo.next;
        }
        System.out.println("Middle value of even size linked list: " + slowTwo.val);

        return;
    }

    /*
     * Fixed Sliding Window Algorithm where window size is predetermined and updated as the array moves
     * O(n)
     * Example Provided: Maximum value within a subarray of size K
     */
    private static void fixedSlidingWindow() {
        int[] input = {0, 1, 2, 3, 4, 5, 6};
        int windowSize = 3;

        if (input.length < windowSize) {
            System.out.println("Window size is larger than array size");
            return;
        }

        int windowSum = 0;
        for(int i = 0; i < windowSize; i++) {
            windowSum = windowSum + input[i];
        }
        int maxSum = windowSum;

        for (int rightPointer = windowSize; rightPointer < input.length; rightPointer++) {
            int leftPointer = rightPointer - windowSize;
            windowSum = windowSum - input[leftPointer] + input[rightPointer];
            maxSum = Math.max(maxSum, windowSum);
        }

        System.out.println("Maximum sum in sliding window: " + maxSum);
        return;
    }

    /*
     * Dynamic Sliding Window Algorithm where a set keeps track of seen items
     * O(n)
     * Example Provided: Longest Substring with no repeating characters
     */
    private static void dynamicSlidingWindowWithSet() {
        String inputString = "abcabcdbb";
        Set<Character> window = new HashSet<>();
        int maxLength = 0;
        int leftPointer = 0;

        for (int rightPointer = 0; rightPointer < inputString.length(); rightPointer++) {
            char rightChar = inputString.charAt(rightPointer);

            while(window.contains(rightChar)) {
                char leftChar = inputString.charAt(leftPointer);
                window.remove(leftChar);
                leftPointer++;
            }

            window.add(rightChar);
            maxLength = Math.max(maxLength, window.size());
        }

        System.out.println("Longest substring with no repeating characters: " + maxLength);
        return;
    }

    /*
     * Dynamic Sliding Window Algorithm where a map is used to track elements
     * O(n)
     * Example Provided: Maximum sum of subarray with at most k distinct elements
     */
    private static void dynamicSlidingWindowWithMap() {
        int[] array = {1, 2, 1, 2, 3};
        int k = 2;
        Map<Integer, Integer> frequency = new HashMap<>();
        int maxSum = 0;
        int currentSum = 0;
        int leftPointer = 0;

        for (int rightPointer = 0; rightPointer < array.length; rightPointer++) {
            currentSum = currentSum + array[rightPointer];
            frequency.put(array[rightPointer], frequency.getOrDefault(array[rightPointer], 0) + 1);

            while (frequency.size() > k) {
                int leftElement = array[leftPointer];
                currentSum = currentSum - leftElement;
                frequency.put(leftElement, frequency.get(leftElement) - 1);
                if (frequency.get(leftElement) == 0) {
                    frequency.remove(leftElement);
                }
                leftPointer++;
            }

            maxSum = Math.max(maxSum, currentSum);
        }

        System.out.println("Maximum length of subarray with " + k + " unique elements: " + maxSum);
        return;
    }

    /*
     * Binary Search Algorithm for searching for an index in a sorted list
     * O(log n) Time Complexity, O(1) Space Complexity
     * Example Provided: Find the index of integer 'target' in a sorted list
     */
    private static void binarySearchWithNumbers() {
        int[] sortedArray = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
        int target = 7;

        int leftPointer = 0;
        int rightPointer = sortedArray.length - 1;

        while (leftPointer <= rightPointer) {
            int mid = leftPointer + (rightPointer - leftPointer) / 2;

            if (sortedArray[mid] == target) {
                System.out.println("Target Index Found: " + mid);
                return;
            }
            else if (sortedArray[mid] < target) {
                leftPointer = mid + 1;
            }
            else {
                rightPointer = mid - 1;
            }
        }

        System.out.println("Target value is not present in the array");
        return;
    }

    /*
     * Binary Search Algorithm for searching conditions such as boolean false->true or array sliced and appended
     * O(log n) Time Complexity, O(1) Space Complexity
     * Example Provided: Given an array of booleans, identify the first instance of 'True'
     */
    private static void binarySearchWithCondition() {
        boolean[] array = {false, false, false, false, false, false, false, true, true, true};
        int leftPointer = 0;
        int rightPointer = array.length - 1;
        int firstTrueIndex = -1;

        while(leftPointer <= rightPointer) {
            int mid = leftPointer + (rightPointer - leftPointer) / 2;

            if (array[mid]) {
                firstTrueIndex = mid;
                rightPointer = mid - 1;
            }
            else {
                leftPointer = mid + 1;
            }
        }

        if (firstTrueIndex == -1) {
            System.out.println("No true index found");
        }

        System.out.println("First True Index: " + firstTrueIndex);
        return;
    }
}
