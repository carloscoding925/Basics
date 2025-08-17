package Java;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
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

    static class ListNode {
        int val;
        ListNode next = null;

        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    static class TreeNode {
        int val;
        List<TreeNode> children;

        TreeNode(int val) {
            this.val = val;
            this.children = new ArrayList<>();
        }

        TreeNode(int val, List<TreeNode> children) {
            this.val = val;
            this.children = children;
        }
    }

    static class GraphNode {
        int val;
        List<GraphNode> neighbors;

        GraphNode(int val) {
            this.val = val;
            this.neighbors = new ArrayList<>();
        }
    }

    public static void main(String[] args) {
        System.out.println("Entry Point");

        arraysAndLists();
        strings();
        sets();
        maps();
        graphsAndTrees();
    }

    private static void arraysAndLists() {
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

        // Two Pointer implementation (slow-fast) pointers (Odd Implementation) (Should land on 2)
        // O(n)
        ListNode linkedList = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4)))));
        ListNode slow = linkedList;
        ListNode fast = linkedList;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        System.out.println("Value at the Middle of Linked List: " + slow.val);

        // Two Pointer implementation (slow-fast) pointers (Even Implementation) (Should land on 3)
        // O(n)
        ListNode linkedListTwo = new ListNode(0, new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5))))));
        ListNode slowTwo = linkedListTwo;
        ListNode fastTwo = linkedListTwo;

        while (fastTwo != null && fastTwo.next != null) {
            fastTwo = fastTwo.next.next;
            slowTwo = slowTwo.next;
        }
        System.out.println("Value at the Middle of Linked List: " + slowTwo.val);

        // Sliding Window Algorithm (Fixed Size)
        // O(n)
        int[] input = {0, 1, 2, 3, 4, 5, 6};
        int windowSize = 3;

        if (input.length < windowSize) {
            System.out.println("Invalid Window Size - Bigger than Input Array");
        }

        int windowSum = 0;
        for (int i = 0; i < windowSize; i++) {
            windowSum = windowSum + input[i];
        }
        int maxSum = windowSum;

        for (int rightPointer = windowSize; rightPointer < input.length; rightPointer++) {
            int leftPointer = rightPointer - windowSize;
            windowSum = windowSum - input[leftPointer] + input[rightPointer];
            maxSum = Math.max(maxSum, windowSum);
        }

        System.out.println("Maximum value in sliding window: " + maxSum);

        // Sliding Window Algorithm with Set (Dynamic Size)
        // O(n)
        // Example: Longest Substring Without Repeating Characters
        String inputString = "abcabcbb";
        Set<Character> window = new HashSet<>();
        int maxLength = 0;
        int swaLeft = 0;

        for (int swaRight = 0; swaRight < inputString.length(); swaRight++) {
            char rightChar = inputString.charAt(swaRight);

            while(window.contains(rightChar)) {
                char leftChar = inputString.charAt(swaLeft);
                window.remove(leftChar);
                swaLeft++;
            }

            window.add(rightChar);
            maxLength = Math.max(maxLength, window.size());
        }

        System.out.println("Longest Substring Without Repeating Characters: " + maxLength);

        // Sliding Window Algorithm with Map (Dynamic Size)
        // O(n)
        // Example: Maximum Sum Subarray with at most k distinct elements
        int[] numsArray = {1, 2, 1, 2, 3};
        int k = 2;
        Map<Integer, Integer> frequency = new HashMap<>();
        int maxArraySum = 0;
        int currentSum = 0;
        int leftPtr = 0;

        for (int rightPtr = 0; rightPtr < numsArray.length; rightPtr++) {
            currentSum = currentSum + numsArray[rightPtr];
            frequency.put(numsArray[rightPtr], frequency.getOrDefault(numsArray[rightPtr], 0) + 1);

            while(frequency.size() > k) {
                int leftElement = numsArray[leftPtr];
                currentSum = currentSum - leftElement;
                frequency.put(leftElement, frequency.get(leftElement) - 1);
                if (frequency.get(leftElement) == 0) {
                    frequency.remove(leftElement);
                }
                leftPtr++;
            }

            maxArraySum = Math.max(maxArraySum, currentSum);
        }

        System.out.println("Maximum sum with at most " + k + " distinct elements: " + maxArraySum);

        // Binary Search Algorithm
        // O(log n) Time Complexity O(1) Space Complexity
        // Sorted Array Example
        int[] sortedArray = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
        int target = 7;

        int bsaLeft = 0;
        int bsaRight = sortedArray.length - 1;

        while(bsaLeft <= bsaRight) {
            int mid = bsaLeft + (bsaRight - bsaLeft) / 2;

            if (sortedArray[mid] == target) {
                System.out.println("Found value index via BSA: " + mid);
                break;
            }
            else if (sortedArray[mid] < target) {
                bsaLeft = mid + 1;
            }
            else {
                bsaRight = mid - 1;
            }
        }

        // Binary Search Algorithm
        // False-True Array Example
        boolean[] boolArray = {false, false, false, false, false, false, false, true, true, true};
        int boolLeft = 0;
        int boolRight = boolArray.length - 1;
        int firstTrueIndex = -1;

        while(boolLeft <= boolRight) {
            int mid = boolLeft + (boolRight - boolLeft) / 2;

            if (boolArray[mid]) {
                firstTrueIndex = mid;
                boolRight = mid - 1;
            }
            else {
                boolLeft = mid + 1;
            }
        }

        System.out.println("First Index with a value of True: " + firstTrueIndex);
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

    private static void graphsAndTrees() {
        // Breadth-First Search on Trees
        // O(n) Time Complexity, O(w) Space Complexity where w is the maximum width

        TreeNode root = new TreeNode(1);
        root.children.add(new TreeNode(2));
        root.children.add(new TreeNode(3));
        root.children.get(0).children.add(new TreeNode(4));
        root.children.get(0).children.add(new TreeNode(5));

        /*
         * This currently throws a warning for dead code, but is required before
         * starting a BFS search
         * 
        if (root == null) {
            System.out.println("Tree is Null");
        }
        */

        Queue<TreeNode> queue = new LinkedList<>();
        int target = 4;

        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.println("Visiting Tree Node: " + node.val);

            for (TreeNode child : node.children) {
                if (child.val == target) {
                    System.out.println("Found Node with Target: " + child.val);
                    break;
                }
                queue.offer(child);
            }
        }

        System.out.println("Tree BFS Search Complete");

        // Depth-First Search on Trees
        // O(n) Time Complexity, O(h) Space Complexity where h is the height of the tree
        /*
         * Use Case:
         * Binary Trees, N-ary Trees
         * When you need to explore all paths or find a specific node
         * When you want to go deep before exploring siblings
        */

        target = 5;
        TreeNode result = dfs(root, 5);
        System.out.println("Found Node with Target Value: " + result.val);
        System.out.println("Tree DFS Complete");

        // Breadth-First Search on Graphs
        // O(V + E) Time Complexity, O(V) Space Complexity where V is vertices and E is edges
        /*
         * Use Case:
         * Grids, Adjacency Lists, Networks
         * Structure contains cycles/duplicate paths
         * Need to find the shortest number of steps
         */

        GraphNode node1 = new GraphNode(1);
        GraphNode node2 = new GraphNode(2);
        GraphNode node3 = new GraphNode(3);
        GraphNode node4 = new GraphNode(4);

        node1.neighbors.add(node2);
        node1.neighbors.add(node3);
        node2.neighbors.add(node1);
        node2.neighbors.add(node4);
        node3.neighbors.add(node1);
        node4.neighbors.add(node2);

        /*
         * This currently throws a warning for dead code, but is required before
         * starting a BFS search
         * 
        if (root == null) {
            System.out.println("Tree is Null");
        }
        */

        Queue<GraphNode> graphQueue = new LinkedList<>();
        Set<GraphNode> visited = new HashSet<>();

        graphQueue.offer(node1);
        visited.add(node1);

        while(!graphQueue.isEmpty()) {
            GraphNode node = graphQueue.poll();
            System.out.println("Visiting graph node: " + node.val);
            
            for (GraphNode neighbor : node.neighbors) {
                if (visited.contains(neighbor)) {
                    continue;
                }
                graphQueue.offer(neighbor);
                visited.add(neighbor);
            }
        }

        System.out.println("Graph BFS Complete");

        // Depth-First Search on Graphs
        // O(V + E) Time Complexity, O(V) Space Complexity where V is vertices and E is edges
        /*
         * Use Case:
         * Exploring all paths, detecting cycles, topological sorting
         * When you want to go as deep as possible before backtracking
         * Connected components, maze solving
        */

        Set<GraphNode> dfsVisited = new HashSet<>();
        dfsGraph(node1, dfsVisited);
        System.out.println("Graph DFS Complete");
    }

    private static TreeNode dfs(TreeNode root, int target) {
        if (root == null) {
            return null;
        }

        if (root.val == target) {
            return root;
        }

        for (TreeNode child : root.children) {
            TreeNode result = dfs(child, target);
            if (result != null) {
                return result;
            }
        }

        return null;
    }

    private static void dfsGraph(GraphNode root, Set<GraphNode> visited) {
        if (root == null) {
            return;
        }

        System.out.println("Visiting graph node via DFS: " + root.val);

        for (GraphNode neighbor : root.neighbors) {
            if (visited.contains(neighbor)) {
                continue;
            }
            visited.add(neighbor);
            dfsGraph(neighbor, visited);
        }
    }
}