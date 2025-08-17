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

    public static void main(String[] args) {
        System.out.println("Entry Point");

        arraysAndLists();
        strings();
        sets();
        maps();
        graphsAndTrees();
        dynamicProgramming();
        sorting();
        advancedTopics();
    }

    static class ListNode {
        int val;
        ListNode next = null;

        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
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

        // Dijkstra's Algorithm - Shortest Path in Weighted Graph
        // O((V + E) log V) Time, O(V) Space
        WeightedGraphNode sourceNode = createWeightedGraph();
        Map<WeightedGraphNode, Integer> shortestDistances = dijkstra(sourceNode);
        System.out.println("Dijkstra's shortest distances from source:");
        for (Map.Entry<WeightedGraphNode, Integer> entry : shortestDistances.entrySet()) {
            System.out.println("Node " + entry.getKey().val + ": " + entry.getValue());
        }
        System.out.println("Dijkstra's Algorithm Complete");

        // Union-Find (Disjoint Set Union) Algorithm
        // O(α(n)) Time for union and find operations (amortized)
        // O(n) Space where α is the inverse Ackermann function (practically constant)
        /*
         * Use Cases:
         * - Detecting cycles in undirected graphs
         * - Finding connected components
         * - Kruskal's minimum spanning tree algorithm
         * - Dynamic connectivity problems
         */
        System.out.println("\nUnion-Find Algorithm:");
        UnionFind uf = new UnionFind(6); // Create union-find for nodes 0-5
        
        // Add some connections (edges)
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        
        System.out.println("After unions (0,1), (1,2), (3,4):");
        System.out.println("Are 0 and 2 connected? " + uf.isConnected(0, 2)); // true
        System.out.println("Are 0 and 3 connected? " + uf.isConnected(0, 3)); // false
        System.out.println("Number of components: " + uf.getComponentCount());
        
        // Connect two components
        uf.union(2, 3);
        System.out.println("\nAfter union (2,3):");
        System.out.println("Are 0 and 4 connected? " + uf.isConnected(0, 4)); // true
        System.out.println("Number of components: " + uf.getComponentCount());
        
        // Demonstrate cycle detection
        int[][] edges = {{0, 1}, {1, 2}, {0, 2}}; // This creates a cycle
        boolean hasCycle = detectCycleUsingUnionFind(edges, 3);
        System.out.println("Graph with edges [(0,1), (1,2), (0,2)] has cycle: " + hasCycle);
        
        System.out.println("Union-Find Algorithm Complete");

        // Prefix Tree (Trie) Implementation
        // O(m) Time for insert, search, and startsWith operations where m is the length of the word
        // O(ALPHABET_SIZE * N * M) Space where N is number of words and M is average length
        /*
         * Use Cases:
         * - Auto-complete/suggestion systems
         * - Spell checkers
         * - IP routing (longest prefix matching)
         * - Dictionary implementations
         * - Word games (finding words with certain prefixes)
         */
        System.out.println("\nPrefix Tree (Trie) Algorithm:");
        Trie trie = new Trie();
        
        // Insert words into the trie
        String[] words = {"apple", "app", "application", "apply", "apartment", "ape"};
        for (String word : words) {
            trie.insert(word);
        }
        System.out.println("Inserted words: " + java.util.Arrays.toString(words));
        
        // Test search functionality
        System.out.println("Search 'app': " + trie.search("app")); // true
        System.out.println("Search 'appl': " + trie.search("appl")); // false
        System.out.println("Search 'apple': " + trie.search("apple")); // true
        System.out.println("Search 'application': " + trie.search("application")); // true
        System.out.println("Search 'banana': " + trie.search("banana")); // false
        
        // Test prefix functionality
        System.out.println("StartsWith 'app': " + trie.startsWith("app")); // true
        System.out.println("StartsWith 'appl': " + trie.startsWith("appl")); // true
        System.out.println("StartsWith 'apt': " + trie.startsWith("apt")); // false
        System.out.println("StartsWith 'a': " + trie.startsWith("a")); // true
        
        // Get all words with a specific prefix
        System.out.println("Words starting with 'app': " + trie.getWordsWithPrefix("app"));
        System.out.println("Words starting with 'apa': " + trie.getWordsWithPrefix("apa"));
        
        // Delete a word
        trie.delete("app");
        System.out.println("After deleting 'app':");
        System.out.println("Search 'app': " + trie.search("app")); // false
        System.out.println("Search 'apple': " + trie.search("apple")); // true (still exists)
        System.out.println("StartsWith 'app': " + trie.startsWith("app")); // true (apple still exists)
        
        System.out.println("Prefix Tree (Trie) Complete");
    }

    static class WeightedGraphNode {
        int val;
        List<Edge> neighbors;

        WeightedGraphNode(int val) {
            this.val = val;
            this.neighbors = new ArrayList<>();
        }
    }

    static class Edge {
        WeightedGraphNode node;
        int weight;

        Edge(WeightedGraphNode node, int weight) {
            this.node = node;
            this.weight = weight;
        }
    }

    private static WeightedGraphNode createWeightedGraph() {
        WeightedGraphNode node1 = new WeightedGraphNode(1);
        WeightedGraphNode node2 = new WeightedGraphNode(2);
        WeightedGraphNode node3 = new WeightedGraphNode(3);
        WeightedGraphNode node4 = new WeightedGraphNode(4);

        // Create weighted edges
        node1.neighbors.add(new Edge(node2, 4));
        node1.neighbors.add(new Edge(node3, 2));
        node2.neighbors.add(new Edge(node4, 3));
        node3.neighbors.add(new Edge(node2, 1));
        node3.neighbors.add(new Edge(node4, 5));

        return node1; // Return source node
    }

    private static Map<WeightedGraphNode, Integer> dijkstra(WeightedGraphNode source) {
        Map<WeightedGraphNode, Integer> distances = new HashMap<>();
        Set<WeightedGraphNode> visited = new HashSet<>();
        
        // Priority queue to always process the node with minimum distance
        Queue<WeightedGraphNode> pq = new java.util.PriorityQueue<>((a, b) -> 
            distances.getOrDefault(a, Integer.MAX_VALUE) - distances.getOrDefault(b, Integer.MAX_VALUE));
        
        // Initialize distances
        distances.put(source, 0);
        pq.offer(source);
        
        while (!pq.isEmpty()) {
            WeightedGraphNode current = pq.poll();
            
            if (visited.contains(current)) {
                continue;
            }
            
            visited.add(current);
            int currentDistance = distances.get(current);
            
            // Check all neighbors
            for (Edge edge : current.neighbors) {
                WeightedGraphNode neighbor = edge.node;
                int newDistance = currentDistance + edge.weight;
                
                // If we found a shorter path, update it
                if (newDistance < distances.getOrDefault(neighbor, Integer.MAX_VALUE)) {
                    distances.put(neighbor, newDistance);
                    pq.offer(neighbor);
                }
            }
        }
        
        return distances;
    }

    // Union-Find (Disjoint Set Union) Data Structure
    static class UnionFind {
        private int[] parent;
        private int[] rank;
        private int componentCount;
        
        public UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            componentCount = n;
            
            // Initialize each node as its own parent (separate component)
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                rank[i] = 0;
            }
        }
        
        // Find operation with path compression
        // O(α(n)) amortized time complexity
        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // Path compression
            }
            return parent[x];
        }
        
        // Union operation with union by rank
        // O(α(n)) amortized time complexity
        public boolean union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            
            if (rootX == rootY) {
                return false; // Already in same component
            }
            
            // Union by rank: attach smaller tree under root of larger tree
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            
            componentCount--;
            return true;
        }
        
        // Check if two nodes are in the same component
        public boolean isConnected(int x, int y) {
            return find(x) == find(y);
        }
        
        // Get the number of connected components
        public int getComponentCount() {
            return componentCount;
        }
        
        // Get the size of the component containing x
        public int getComponentSize(int x) {
            int root = find(x);
            int size = 0;
            for (int i = 0; i < parent.length; i++) {
                if (find(i) == root) {
                    size++;
                }
            }
            return size;
        }
    }

    // Prefix Tree (Trie) Data Structure
    static class Trie {
        private TrieNode root;
        
        // TrieNode class representing each node in the trie
        static class TrieNode {
            private TrieNode[] children;
            private boolean isEndOfWord;
            private static final int ALPHABET_SIZE = 26; // for lowercase letters a-z
            
            public TrieNode() {
                children = new TrieNode[ALPHABET_SIZE];
                isEndOfWord = false;
            }
        }
        
        public Trie() {
            root = new TrieNode();
        }
        
        // Insert a word into the trie
        // O(m) Time Complexity where m is the length of the word
        public void insert(String word) {
            if (word == null || word.isEmpty()) {
                return;
            }
            
            TrieNode current = root;
            for (char ch : word.toCharArray()) {
                int index = ch - 'a'; // Convert char to index (0-25)
                
                if (current.children[index] == null) {
                    current.children[index] = new TrieNode();
                }
                
                current = current.children[index];
            }
            
            current.isEndOfWord = true;
        }
        
        // Search for a complete word in the trie
        // O(m) Time Complexity where m is the length of the word
        public boolean search(String word) {
            if (word == null || word.isEmpty()) {
                return false;
            }
            
            TrieNode current = root;
            for (char ch : word.toCharArray()) {
                int index = ch - 'a';
                
                if (current.children[index] == null) {
                    return false;
                }
                
                current = current.children[index];
            }
            
            return current.isEndOfWord;
        }
        
        // Check if any word in the trie starts with the given prefix
        // O(m) Time Complexity where m is the length of the prefix
        public boolean startsWith(String prefix) {
            if (prefix == null || prefix.isEmpty()) {
                return true;
            }
            
            TrieNode current = root;
            for (char ch : prefix.toCharArray()) {
                int index = ch - 'a';
                
                if (current.children[index] == null) {
                    return false;
                }
                
                current = current.children[index];
            }
            
            return true;
        }
        
        // Get all words that start with the given prefix
        public List<String> getWordsWithPrefix(String prefix) {
            List<String> result = new ArrayList<>();
            
            if (prefix == null) {
                return result;
            }
            
            TrieNode current = root;
            
            // Navigate to the end of the prefix
            for (char ch : prefix.toCharArray()) {
                int index = ch - 'a';
                
                if (current.children[index] == null) {
                    return result; // No words with this prefix
                }
                
                current = current.children[index];
            }
            
            // Perform DFS to find all words starting from this node
            dfsCollectWords(current, prefix, result);
            return result;
        }
        
        // Helper method for DFS to collect all words from a given node
        private void dfsCollectWords(TrieNode node, String currentWord, List<String> result) {
            if (node.isEndOfWord) {
                result.add(currentWord);
            }
            
            for (int i = 0; i < TrieNode.ALPHABET_SIZE; i++) {
                if (node.children[i] != null) {
                    char nextChar = (char) ('a' + i);
                    dfsCollectWords(node.children[i], currentWord + nextChar, result);
                }
            }
        }
        
        // Delete a word from the trie
        // O(m) Time Complexity where m is the length of the word
        public boolean delete(String word) {
            if (word == null || word.isEmpty()) {
                return false;
            }
            
            return deleteHelper(root, word, 0);
        }
        
        // Helper method for deletion with recursion
        private boolean deleteHelper(TrieNode current, String word, int index) {
            if (index == word.length()) {
                // We've reached the end of the word
                if (!current.isEndOfWord) {
                    return false; // Word doesn't exist
                }
                
                current.isEndOfWord = false;
                
                // Return true if current node has no children (can be deleted)
                return !hasChildren(current);
            }
            
            char ch = word.charAt(index);
            int charIndex = ch - 'a';
            TrieNode node = current.children[charIndex];
            
            if (node == null) {
                return false; // Word doesn't exist
            }
            
            boolean shouldDeleteChild = deleteHelper(node, word, index + 1);
            
            if (shouldDeleteChild) {
                current.children[charIndex] = null;
                
                // Return true if current node has no children and is not end of another word
                return !current.isEndOfWord && !hasChildren(current);
            }
            
            return false;
        }
        
        // Helper method to check if a node has any children
        private boolean hasChildren(TrieNode node) {
            for (TrieNode child : node.children) {
                if (child != null) {
                    return true;
                }
            }
            return false;
        }
        
        // Get the total number of words in the trie
        public int countWords() {
            return countWordsHelper(root);
        }
        
        // Helper method to count words recursively
        private int countWordsHelper(TrieNode node) {
            int count = 0;
            
            if (node.isEndOfWord) {
                count = 1;
            }
            
            for (TrieNode child : node.children) {
                if (child != null) {
                    count += countWordsHelper(child);
                }
            }
            
            return count;
        }
        
        // Check if the trie is empty
        public boolean isEmpty() {
            return countWords() == 0;
        }
    }

    // Helper method: Detect cycle in undirected graph using Union-Find
    private static boolean detectCycleUsingUnionFind(int[][] edges, int n) {
        UnionFind uf = new UnionFind(n);
        
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            
            // If both vertices are already in same component, adding this edge creates a cycle
            if (uf.isConnected(u, v)) {
                return true;
            }
            
            uf.union(u, v);
        }
        
        return false;
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

    private static void dynamicProgramming() {
        // Fibonacci Sequence - Classic DP Example
        // O(n) Time, O(1) Space (optimized)
        int n = 10;
        int fibResult = fibonacci(n);
        System.out.println("Fibonacci of " + n + ": " + fibResult);

        // Climbing Stairs - How many ways to reach step n
        // O(n) Time, O(1) Space
        int steps = 5;
        int waysToClimb = climbStairs(steps);
        System.out.println("Ways to climb " + steps + " stairs: " + waysToClimb);

        // Coin Change - Minimum coins needed to make amount
        // O(amount * coins.length) Time, O(amount) Space
        int[] coins = {1, 3, 4};
        int amount = 6;
        int minCoins = coinChange(coins, amount);
        System.out.println("Minimum coins to make " + amount + ": " + minCoins);

        // Longest Common Subsequence
        // O(m * n) Time, O(m * n) Space
        String text1 = "abcde";
        String text2 = "ace";
        int lcsLength = longestCommonSubsequence(text1, text2);
        System.out.println("LCS length of '" + text1 + "' and '" + text2 + "': " + lcsLength);

        // Maximum Subarray Sum (Kadane's Algorithm)
        // O(n) Time, O(1) Space
        int[] array = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        int maxSum = maxSubarraySum(array);
        System.out.println("Maximum subarray sum: " + maxSum);

        // House Robber - Maximum money without robbing adjacent houses
        // O(n) Time, O(1) Space
        int[] houses = {2, 7, 9, 3, 1};
        int maxRobbed = rob(houses);
        System.out.println("Maximum money robbed: " + maxRobbed);

        System.out.println("Dynamic Programming Complete");
    }

    private static int fibonacci(int n) {
        if (n <= 1) return n;
        
        int prev2 = 0, prev1 = 1;
        for (int i = 2; i <= n; i++) {
            int current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    }

    private static int climbStairs(int n) {
        if (n <= 2) return n;
        
        int oneStep = 1, twoSteps = 2;
        for (int i = 3; i <= n; i++) {
            int current = oneStep + twoSteps;
            oneStep = twoSteps;
            twoSteps = current;
        }
        return twoSteps;
    }

    private static int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        java.util.Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    private static int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    private static int maxSubarraySum(int[] nums) {
        int maxSoFar = nums[0];
        int maxEndingHere = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        return maxSoFar;
    }

    private static int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int prev2 = 0, prev1 = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int current = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    }

    private static void sorting() {
        int[] testArray = {64, 34, 25, 12, 22, 11, 90, 5};
        System.out.println("Original array: " + java.util.Arrays.toString(testArray));
        
        // Bubble Sort - O(n^2) Time, O(1) Space
        // Simple but inefficient, compares adjacent elements
        int[] bubbleArray = testArray.clone();
        bubbleSort(bubbleArray);
        System.out.println("Bubble Sort: " + java.util.Arrays.toString(bubbleArray));
        
        // Selection Sort - O(n^2) Time, O(1) Space
        // Finds minimum element and places it at the beginning
        int[] selectionArray = testArray.clone();
        selectionSort(selectionArray);
        System.out.println("Selection Sort: " + java.util.Arrays.toString(selectionArray));
        
        // Insertion Sort - O(n^2) Time, O(1) Space
        // Builds sorted array one element at a time, good for small arrays
        int[] insertionArray = testArray.clone();
        insertionSort(insertionArray);
        System.out.println("Insertion Sort: " + java.util.Arrays.toString(insertionArray)); 

        // Merge Sort - O(n log n) Time, O(n) Space
        // Divide and conquer, stable sort, good for large datasets
        int[] mergeArray = testArray.clone();
        mergeSort(mergeArray, 0, mergeArray.length - 1);
        System.out.println("Merge Sort: " + java.util.Arrays.toString(mergeArray));
        
        // Quick Sort - O(n log n) average, O(n^2) worst case Time, O(log n) Space
        // Divide and conquer with pivot, fastest in practice
        int[] quickArray = testArray.clone();
        quickSort(quickArray, 0, quickArray.length - 1);
        System.out.println("Quick Sort: " + java.util.Arrays.toString(quickArray));
        
        // Heap Sort - O(n log n) Time, O(1) Space
        // Uses binary heap data structure, not stable but in-place
        int[] heapArray = testArray.clone();
        heapSort(heapArray);
        System.out.println("Heap Sort: " + java.util.Arrays.toString(heapArray));
        
        // Counting Sort - O(n + k) Time, O(k) Space where k is range of input
        // Non-comparison sort, only works with integers in small range
        int[] countingArray = {4, 2, 2, 8, 3, 3, 1};
        int[] countingSorted = countingSort(countingArray);
        System.out.println("Counting Sort: " + java.util.Arrays.toString(countingSorted));
        
        System.out.println("Sorting Complete");
    }

    // Bubble Sort - Simple but inefficient
    private static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // Swap elements
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapped = true;
                }
            }
            // Optimization: if no swapping occurred, array is sorted
            if (!swapped) break;
        }
    }

    // Selection Sort - Find minimum and place at beginning
    private static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            // Swap minimum element with first element
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }

    // Insertion Sort - Build sorted array one element at a time
    private static void insertionSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int key = arr[i];
            int j = i - 1;
            
            // Move elements greater than key one position ahead
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    // Merge Sort - Divide and conquer approach
    private static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            
            // Recursively sort both halves
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            
            // Merge the sorted halves
            merge(arr, left, mid, right);
        }
    }

    private static void merge(int[] arr, int left, int mid, int right) {
        // Create temporary arrays for left and right subarrays
        int leftSize = mid - left + 1;
        int rightSize = right - mid;
        
        int[] leftArray = new int[leftSize];
        int[] rightArray = new int[rightSize];
        
        // Copy data to temporary arrays
        for (int i = 0; i < leftSize; i++) {
            leftArray[i] = arr[left + i];
        }
        for (int j = 0; j < rightSize; j++) {
            rightArray[j] = arr[mid + 1 + j];
        }
        
        // Merge the temporary arrays back into arr[left..right]
        int i = 0, j = 0, k = left;
        
        while (i < leftSize && j < rightSize) {
            if (leftArray[i] <= rightArray[j]) {
                arr[k] = leftArray[i];
                i++;
            } else {
                arr[k] = rightArray[j];
                j++;
            }
            k++;
        }
        
        // Copy remaining elements
        while (i < leftSize) {
            arr[k] = leftArray[i];
            i++;
            k++;
        }
        while (j < rightSize) {
            arr[k] = rightArray[j];
            j++;
            k++;
        }
    }

    // Quick Sort - Divide and conquer with pivot
    private static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(arr, low, high);
            
            // Recursively sort elements before and after partition
            quickSort(arr, low, pivotIndex - 1);
            quickSort(arr, pivotIndex + 1, high);
        }
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high]; // Choose last element as pivot
        int i = low - 1; // Index of smaller element
        
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                // Swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        // Swap arr[i+1] and arr[high] (pivot)
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        return i + 1;
    }

    // Heap Sort - Uses binary heap data structure
    private static void heapSort(int[] arr) {
        int n = arr.length;
        
        // Build heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // Extract elements from heap one by one
        for (int i = n - 1; i > 0; i--) {
            // Move current root to end
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            
            // Call heapify on the reduced heap
            heapify(arr, i, 0);
        }
    }

    private static void heapify(int[] arr, int n, int i) {
        int largest = i; // Initialize largest as root
        int left = 2 * i + 1; // Left child
        int right = 2 * i + 2; // Right child
        
        // If left child is larger than root
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        // If right child is larger than largest so far
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        // If largest is not root
        if (largest != i) {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;
            
            // Recursively heapify the affected sub-tree
            heapify(arr, n, largest);
        }
    }

    // Counting Sort - Non-comparison sort for integers
    private static int[] countingSort(int[] arr) {
        // Find the maximum element to determine range
        int max = java.util.Arrays.stream(arr).max().orElse(0);
        int min = java.util.Arrays.stream(arr).min().orElse(0);
        int range = max - min + 1;
        
        // Create count array and output array
        int[] count = new int[range];
        int[] output = new int[arr.length];
        
        // Count occurrences of each element
        for (int value : arr) {
            count[value - min]++;
        }
        
        // Modify count array to store actual positions
        for (int i = 1; i < range; i++) {
            count[i] += count[i - 1];
        }
        
        // Build output array
        for (int i = arr.length - 1; i >= 0; i--) {
            output[count[arr[i] - min] - 1] = arr[i];
            count[arr[i] - min]--;
        }
        
        return output;
    }

    private static void advancedTopics() {
        // LRU Cache Implementation
        // O(1) Time for get() and put() operations
        // Combines HashMap for O(1) access and Doubly Linked List for O(1) insertion/deletion
        System.out.println("Testing LRU Cache:");
        
        LRUCache cache = new LRUCache(3); // Capacity of 3
        
        // Add some values
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        System.out.println("Added (1,10), (2,20), (3,30)");
        
        // Access key 1 (makes it most recently used)
        System.out.println("Get key 1: " + cache.get(1)); // Returns 10
        
        // Add new key-value pair (should evict key 2 as it's least recently used)
        cache.put(4, 40);
        System.out.println("Added (4,40) - should evict key 2");
        
        // Try to access evicted key
        System.out.println("Get key 2: " + cache.get(2)); // Returns -1 (not found)
        
        // Access remaining keys
        System.out.println("Get key 1: " + cache.get(1)); // Returns 10
        System.out.println("Get key 3: " + cache.get(3)); // Returns 30
        System.out.println("Get key 4: " + cache.get(4)); // Returns 40
        
        System.out.println("LRU Cache Complete");
        
        // Heap/Priority Queue Implementation
        // Min Heap - O(log n) insertion/deletion, O(1) peek
        // Max Heap - O(log n) insertion/deletion, O(1) peek
        System.out.println("\nTesting Min Heap:");
        
        MinHeap minHeap = new MinHeap(10);
        int[] values = {4, 10, 3, 5, 1, 8, 2};
        
        // Insert values into min heap
        for (int val : values) {
            minHeap.insert(val);
        }
        System.out.println("Inserted values: " + java.util.Arrays.toString(values));
        System.out.println("Min Heap: " + minHeap.toString());
        System.out.println("Min element (peek): " + minHeap.peek());
        
        // Extract min elements
        System.out.println("Extracting minimums:");
        while (!minHeap.isEmpty()) {
            System.out.print(minHeap.extractMin() + " ");
        }
        System.out.println();
        
        System.out.println("\nTesting Max Heap:");
        
        MaxHeap maxHeap = new MaxHeap(10);
        
        // Insert values into max heap
        for (int val : values) {
            maxHeap.insert(val);
        }
        System.out.println("Inserted values: " + java.util.Arrays.toString(values));
        System.out.println("Max Heap: " + maxHeap.toString());
        System.out.println("Max element (peek): " + maxHeap.peek());
        
        // Extract max elements
        System.out.println("Extracting maximums:");
        while (!maxHeap.isEmpty()) {
            System.out.print(maxHeap.extractMax() + " ");
        }
        System.out.println();
        
        // Priority Queue Use Case: Find K Largest Elements
        System.out.println("\nFinding K Largest Elements using Min Heap:");
        int[] array = {3, 2, 1, 5, 6, 4};
        int k = 3;
        int[] kLargest = findKLargest(array, k);
        System.out.println("Array: " + java.util.Arrays.toString(array));
        System.out.println("K=" + k + " largest elements: " + java.util.Arrays.toString(kLargest));
        
        System.out.println("\nHeap/Priority Queue Complete");
    }
    
    // LRU Cache Implementation using HashMap + Doubly Linked List
    static class LRUCache {
        private int capacity;
        private Map<Integer, Node> cache;
        private Node head;
        private Node tail;
        
        // Doubly Linked List Node
        static class Node {
            int key;
            int value;
            Node prev;
            Node next;
            
            Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }
        
        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.cache = new HashMap<>();
            
            // Create dummy head and tail nodes
            this.head = new Node(0, 0);
            this.tail = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
        }
        
        public int get(int key) {
            Node node = cache.get(key);
            if (node == null) {
                return -1; // Key not found
            }
            
            // Move accessed node to head (most recently used)
            moveToHead(node);
            return node.value;
        }
        
        public void put(int key, int value) {
            Node node = cache.get(key);
            
            if (node != null) {
                // Update existing node
                node.value = value;
                moveToHead(node);
            } else {
                // Add new node
                Node newNode = new Node(key, value);
                
                if (cache.size() >= capacity) {
                    // Remove least recently used node (tail)
                    Node tail = removeTail();
                    cache.remove(tail.key);
                }
                
                cache.put(key, newNode);
                addToHead(newNode);
            }
        }
        
        // Helper method to add node right after head
        private void addToHead(Node node) {
            node.prev = head;
            node.next = head.next;
            head.next.prev = node;
            head.next = node;
        }
        
        // Helper method to remove a node from the list
        private void removeNode(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }
        
        // Helper method to move node to head
        private void moveToHead(Node node) {
            removeNode(node);
            addToHead(node);
        }
        
        // Helper method to remove tail node
        private Node removeTail() {
            Node lastNode = tail.prev;
            removeNode(lastNode);
            return lastNode;
        }
    }
    
    // Min Heap Implementation using Array
    static class MinHeap {
        private int[] heap;
        private int size;
        private int capacity;
        
        public MinHeap(int capacity) {
            this.capacity = capacity;
            this.size = 0;
            this.heap = new int[capacity];
        }
        
        // Get parent index
        private int parent(int i) {
            return (i - 1) / 2;
        }
        
        // Get left child index
        private int leftChild(int i) {
            return 2 * i + 1;
        }
        
        // Get right child index
        private int rightChild(int i) {
            return 2 * i + 2;
        }
        
        // Check if heap is empty
        public boolean isEmpty() {
            return size == 0;
        }
        
        // Get minimum element (root)
        public int peek() {
            if (isEmpty()) {
                throw new RuntimeException("Heap is empty");
            }
            return heap[0];
        }
        
        // Insert element into heap
        public void insert(int value) {
            if (size >= capacity) {
                throw new RuntimeException("Heap is full");
            }
            
            // Insert at the end
            heap[size] = value;
            size++;
            
            // Heapify up
            heapifyUp(size - 1);
        }
        
        // Extract minimum element
        public int extractMin() {
            if (isEmpty()) {
                throw new RuntimeException("Heap is empty");
            }
            
            int min = heap[0];
            
            // Move last element to root
            heap[0] = heap[size - 1];
            size--;
            
            // Heapify down
            if (size > 0) {
                heapifyDown(0);
            }
            
            return min;
        }
        
        // Heapify up (bubble up)
        private void heapifyUp(int index) {
            while (index > 0 && heap[parent(index)] > heap[index]) {
                swap(index, parent(index));
                index = parent(index);
            }
        }
        
        // Heapify down (bubble down)
        private void heapifyDown(int index) {
            int smallest = index;
            int left = leftChild(index);
            int right = rightChild(index);
            
            // Find smallest among index, left child, and right child
            if (left < size && heap[left] < heap[smallest]) {
                smallest = left;
            }
            
            if (right < size && heap[right] < heap[smallest]) {
                smallest = right;
            }
            
            // If smallest is not the current index, swap and continue
            if (smallest != index) {
                swap(index, smallest);
                heapifyDown(smallest);
            }
        }
        
        // Swap two elements in heap
        private void swap(int i, int j) {
            int temp = heap[i];
            heap[i] = heap[j];
            heap[j] = temp;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < size; i++) {
                sb.append(heap[i]);
                if (i < size - 1) sb.append(", ");
            }
            sb.append("]");
            return sb.toString();
        }
    }
    
    // Max Heap Implementation using Array
    static class MaxHeap {
        private int[] heap;
        private int size;
        private int capacity;
        
        public MaxHeap(int capacity) {
            this.capacity = capacity;
            this.size = 0;
            this.heap = new int[capacity];
        }
        
        // Get parent index
        private int parent(int i) {
            return (i - 1) / 2;
        }
        
        // Get left child index
        private int leftChild(int i) {
            return 2 * i + 1;
        }
        
        // Get right child index
        private int rightChild(int i) {
            return 2 * i + 2;
        }
        
        // Check if heap is empty
        public boolean isEmpty() {
            return size == 0;
        }
        
        // Get maximum element (root)
        public int peek() {
            if (isEmpty()) {
                throw new RuntimeException("Heap is empty");
            }
            return heap[0];
        }
        
        // Insert element into heap
        public void insert(int value) {
            if (size >= capacity) {
                throw new RuntimeException("Heap is full");
            }
            
            // Insert at the end
            heap[size] = value;
            size++;
            
            // Heapify up
            heapifyUp(size - 1);
        }
        
        // Extract maximum element
        public int extractMax() {
            if (isEmpty()) {
                throw new RuntimeException("Heap is empty");
            }
            
            int max = heap[0];
            
            // Move last element to root
            heap[0] = heap[size - 1];
            size--;
            
            // Heapify down
            if (size > 0) {
                heapifyDown(0);
            }
            
            return max;
        }
        
        // Heapify up (bubble up)
        private void heapifyUp(int index) {
            while (index > 0 && heap[parent(index)] < heap[index]) {
                swap(index, parent(index));
                index = parent(index);
            }
        }
        
        // Heapify down (bubble down)
        private void heapifyDown(int index) {
            int largest = index;
            int left = leftChild(index);
            int right = rightChild(index);
            
            // Find largest among index, left child, and right child
            if (left < size && heap[left] > heap[largest]) {
                largest = left;
            }
            
            if (right < size && heap[right] > heap[largest]) {
                largest = right;
            }
            
            // If largest is not the current index, swap and continue
            if (largest != index) {
                swap(index, largest);
                heapifyDown(largest);
            }
        }
        
        // Swap two elements in heap
        private void swap(int i, int j) {
            int temp = heap[i];
            heap[i] = heap[j];
            heap[j] = temp;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < size; i++) {
                sb.append(heap[i]);
                if (i < size - 1) sb.append(", ");
            }
            sb.append("]");
            return sb.toString();
        }
    }
    
    // Helper method: Find K Largest Elements using Min Heap
    private static int[] findKLargest(int[] nums, int k) {
        MinHeap minHeap = new MinHeap(k);
        
        // Process each element
        for (int num : nums) {
            if (minHeap.size < k) {
                minHeap.insert(num);
            } else if (num > minHeap.peek()) {
                minHeap.extractMin();
                minHeap.insert(num);
            }
        }
        
        // Extract all elements from heap
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = minHeap.extractMin();
        }
        
        return result;
    }
}