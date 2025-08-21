package Java.Algorithms;

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

        strings();
        sets();
        maps();
        graphsAndTrees();
        dynamicProgramming();
        sorting();
        advancedTopics();
        stacks();
        queues();
        bitManipulation();
        greedy();
    }

    static class ListNode {
        int val;
        ListNode next = null;

        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
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

        // Backtracking Algorithm Implementation
        // Time complexity varies by problem - often O(b^d) where b is branching factor and d is depth
        // Space complexity O(d) for recursion stack depth
        /*
         * Use Cases:
         * - N-Queens problem, Sudoku solver, Knight's tour
         * - Generating permutations and combinations
         * - Maze solving, pathfinding with constraints
         * - Constraint satisfaction problems
         */
        System.out.println("\nBacktracking Algorithms:");
        
        // Example 1: Generate all permutations of an array
        int[] nums = {1, 2, 3};
        System.out.println("Generating permutations of: " + java.util.Arrays.toString(nums));
        List<List<Integer>> permutations = generatePermutations(nums);
        System.out.println("All permutations:");
        for (List<Integer> perm : permutations) {
            System.out.println(perm);
        }
        
        // Example 2: N-Queens problem (4x4 board)
        int n = 4;
        System.out.println("\nSolving " + n + "-Queens problem:");
        List<List<String>> queenSolutions = solveNQueens(n);
        System.out.println("Number of solutions: " + queenSolutions.size());
        if (!queenSolutions.isEmpty()) {
            System.out.println("First solution:");
            for (String row : queenSolutions.get(0)) {
                System.out.println(row);
            }
        }
        
        // Example 3: Subset generation
        int[] subsetNums = {1, 2, 3};
        System.out.println("\nGenerating all subsets of: " + java.util.Arrays.toString(subsetNums));
        List<List<Integer>> subsets = generateSubsets(subsetNums);
        System.out.println("All subsets:");
        for (List<Integer> subset : subsets) {
            System.out.println(subset);
        }
        
        // Example 4: Combination Sum
        int[] candidates = {2, 3, 6, 7};
        int target = 7;
        System.out.println("\nFinding combinations that sum to " + target + " using: " + java.util.Arrays.toString(candidates));
        List<List<Integer>> combinations = combinationSum(candidates, target);
        System.out.println("Valid combinations:");
        for (List<Integer> combination : combinations) {
            System.out.println(combination);
        }
        
        System.out.println("\nBacktracking Complete");
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

    // Backtracking Algorithm Implementations
    
    // Generate all permutations of an array
    // O(n! * n) Time, O(n) Space (not counting output)
    private static List<List<Integer>> generatePermutations(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        
        backtrackPermutations(nums, current, used, result);
        return result;
    }
    
    private static void backtrackPermutations(int[] nums, List<Integer> current, boolean[] used, List<List<Integer>> result) {
        // Base case: if current permutation is complete
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current)); // Make a copy
            return;
        }
        
        // Try each unused number
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                // Choose
                current.add(nums[i]);
                used[i] = true;
                
                // Explore
                backtrackPermutations(nums, current, used, result);
                
                // Unchoose (backtrack)
                current.remove(current.size() - 1);
                used[i] = false;
            }
        }
    }
    
    // N-Queens Problem
    // O(N!) Time, O(N) Space
    private static List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        int[] queens = new int[n]; // queens[i] = column position of queen in row i
        
        backtrackNQueens(0, n, queens, result);
        return result;
    }
    
    private static void backtrackNQueens(int row, int n, int[] queens, List<List<String>> result) {
        // Base case: all queens placed
        if (row == n) {
            result.add(constructBoard(queens, n));
            return;
        }
        
        // Try placing queen in each column of current row
        for (int col = 0; col < n; col++) {
            if (isSafePosition(row, col, queens)) {
                // Choose
                queens[row] = col;
                
                // Explore
                backtrackNQueens(row + 1, n, queens, result);
                
                // Unchoose (implicit - we'll overwrite queens[row] in next iteration)
            }
        }
    }
    
    private static boolean isSafePosition(int row, int col, int[] queens) {
        // Check if placing queen at (row, col) conflicts with existing queens
        for (int i = 0; i < row; i++) {
            int existingCol = queens[i];
            
            // Check column conflict
            if (existingCol == col) {
                return false;
            }
            
            // Check diagonal conflicts
            if (Math.abs(row - i) == Math.abs(col - existingCol)) {
                return false;
            }
        }
        return true;
    }
    
    private static List<String> constructBoard(int[] queens, int n) {
        List<String> board = new ArrayList<>();
        
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (queens[i] == j) {
                    row.append('Q');
                } else {
                    row.append('.');
                }
            }
            board.add(row.toString());
        }
        
        return board;
    }
    
    // Generate all subsets (power set)
    // O(2^n * n) Time, O(n) Space (not counting output)
    private static List<List<Integer>> generateSubsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        
        backtrackSubsets(nums, 0, current, result);
        return result;
    }
    
    private static void backtrackSubsets(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
        // Add current subset to result (every recursive call generates a valid subset)
        result.add(new ArrayList<>(current));
        
        // Try adding each remaining element
        for (int i = start; i < nums.length; i++) {
            // Choose
            current.add(nums[i]);
            
            // Explore (move to next index to avoid duplicates)
            backtrackSubsets(nums, i + 1, current, result);
            
            // Unchoose (backtrack)
            current.remove(current.size() - 1);
        }
    }
    
    // Combination Sum - find all combinations that sum to target
    // O(2^target) Time in worst case, O(target) Space
    private static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        
        // Sort to enable early termination
        java.util.Arrays.sort(candidates);
        
        backtrackCombinationSum(candidates, target, 0, current, result);
        return result;
    }
    
    private static void backtrackCombinationSum(int[] candidates, int remainingTarget, int start, 
                                               List<Integer> current, List<List<Integer>> result) {
        // Base case: found valid combination
        if (remainingTarget == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        // Early termination: if remaining target is negative
        if (remainingTarget < 0) {
            return;
        }
        
        // Try each candidate starting from 'start' index
        for (int i = start; i < candidates.length; i++) {
            // Early termination: if current candidate is too large
            if (candidates[i] > remainingTarget) {
                break; // Since array is sorted, all remaining candidates are also too large
            }
            
            // Choose
            current.add(candidates[i]);
            
            // Explore (note: we pass 'i' not 'i+1' to allow reusing same number)
            backtrackCombinationSum(candidates, remainingTarget - candidates[i], i, current, result);
            
            // Unchoose (backtrack)
            current.remove(current.size() - 1);
        }
    }

    private static void stacks() {
        // Stack Data Structure - LIFO (Last In, First Out)
        // O(1) for push, pop, peek operations
        /*
         * Use Cases:
         * - Function call management (call stack)
         * - Undo operations in applications
         * - Expression evaluation and syntax parsing
         * - Backtracking algorithms
         * - Browser history navigation
         */
        System.out.println("Stack Algorithms:");
        
        // Example 1: Valid Parentheses
        // O(n) Time, O(n) Space
        String[] testExpressions = {"()", "()[]{}", "(]", "([)]", "{[]}"};
        System.out.println("Testing Valid Parentheses:");
        for (String expr : testExpressions) {
            boolean isValid = isValidParentheses(expr);
            System.out.println("'" + expr + "' is valid: " + isValid);
        }
        
        // Example 2: Evaluate Reverse Polish Notation (RPN)
        // O(n) Time, O(n) Space
        String[] rpnTokens = {"2", "1", "+", "3", "*"}; // ((2 + 1) * 3) = 9
        int rpnResult = evaluateRPN(rpnTokens);
        System.out.println("RPN evaluation of " + java.util.Arrays.toString(rpnTokens) + " = " + rpnResult);
        
        // Example 3: Daily Temperatures (Next Greater Element)
        // O(n) Time, O(n) Space
        int[] temperatures = {73, 74, 75, 71, 69, 72, 76, 73};
        int[] waitDays = dailyTemperatures(temperatures);
        System.out.println("Temperatures: " + java.util.Arrays.toString(temperatures));
        System.out.println("Days to wait for warmer weather: " + java.util.Arrays.toString(waitDays));
        
        // Example 4: Largest Rectangle in Histogram
        // O(n) Time, O(n) Space
        int[] heights = {2, 1, 5, 6, 2, 3};
        int maxArea = largestRectangleArea(heights);
        System.out.println("Heights: " + java.util.Arrays.toString(heights));
        System.out.println("Largest rectangle area: " + maxArea);
        
        // Example 5: Simplify Path (Unix-style path)
        // O(n) Time, O(n) Space
        String[] paths = {"/home/", "/a/./b/../../c/", "/a/../../b/../c//.//", "/a//b////c/d//././/.."};
        System.out.println("Path Simplification:");
        for (String path : paths) {
            String simplified = simplifyPath(path);
            System.out.println("'" + path + "' -> '" + simplified + "'");
        }
        
        System.out.println("Stack Algorithms Complete");
    }

    private static void queues() {
        // Queue Data Structure - FIFO (First In, First Out)
        // O(1) for offer/enqueue, poll/dequeue, peek operations
        /*
         * Use Cases:
         * - Task scheduling and process management
         * - Breadth-First Search (BFS) traversal
         * - Level-order tree traversal
         * - Handling requests in web servers
         * - Buffer for data streams
         */
        System.out.println("\nQueue Algorithms:");
        
        // Example 1: Binary Tree Level Order Traversal
        // O(n) Time, O(w) Space where w is maximum width
        System.out.println("Binary Tree Level Order Traversal:");
        BinaryTreeNode root = createSampleBinaryTree();
        List<List<Integer>> levelOrder = levelOrderTraversal(root);
        System.out.println("Level order traversal: " + levelOrder);
        
        // Example 2: Sliding Window Maximum
        // O(n) Time, O(k) Space using deque
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        int k = 3;
        int[] maxSlidingWindow = maxSlidingWindow(nums, k);
        System.out.println("Array: " + java.util.Arrays.toString(nums));
        System.out.println("Sliding window maximum (k=" + k + "): " + java.util.Arrays.toString(maxSlidingWindow));
        
        // Example 3: First Unique Character in String
        // O(n) Time, O(1) Space (limited alphabet)
        String[] testStrings = {"leetcode", "loveleetcode", "aabb"};
        System.out.println("First Unique Character:");
        for (String s : testStrings) {
            int firstUnique = firstUniqueChar(s);
            if (firstUnique != -1) {
                System.out.println("'" + s + "': first unique char is '" + s.charAt(firstUnique) + "' at index " + firstUnique);
            } else {
                System.out.println("'" + s + "': no unique character found");
            }
        }
        
        // Example 4: Design Hit Counter
        // O(1) amortized for hit(), O(300) for getHits()
        System.out.println("Hit Counter Design:");
        HitCounter hitCounter = new HitCounter();
        hitCounter.hit(1);
        hitCounter.hit(2);
        hitCounter.hit(3);
        System.out.println("Hits at timestamp 4: " + hitCounter.getHits(4)); // Should be 3
        hitCounter.hit(300);
        System.out.println("Hits at timestamp 300: " + hitCounter.getHits(300)); // Should be 4
        System.out.println("Hits at timestamp 301: " + hitCounter.getHits(301)); // Should be 3 (first hit expired)
        
        // Example 5: Number of Islands (BFS approach)
        // O(m*n) Time, O(min(m,n)) Space
        char[][] grid = {
            {'1','1','1','1','0'},
            {'1','1','0','1','0'},
            {'1','1','0','0','0'},
            {'0','0','0','0','0'}
        };
        int numIslands = numIslands(grid);
        System.out.println("Number of islands in grid: " + numIslands);
        
        System.out.println("Queue Algorithms Complete");
    }

    // Stack Algorithm Implementations
    
    // Valid Parentheses - Check if parentheses are properly balanced
    // O(n) Time, O(n) Space
    private static boolean isValidParentheses(String s) {
        java.util.Stack<Character> stack = new java.util.Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else if (c == ')' || c == ']' || c == '}') {
                if (stack.isEmpty()) {
                    return false;
                }
                
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
    
    // Evaluate Reverse Polish Notation
    // O(n) Time, O(n) Space
    private static int evaluateRPN(String[] tokens) {
        java.util.Stack<Integer> stack = new java.util.Stack<>();
        
        for (String token : tokens) {
            if (token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                
                switch (token) {
                    case "+": stack.push(a + b); break;
                    case "-": stack.push(a - b); break;
                    case "*": stack.push(a * b); break;
                    case "/": stack.push(a / b); break;
                }
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        
        return stack.pop();
    }
    
    // Daily Temperatures - Find next warmer temperature
    // O(n) Time, O(n) Space
    private static int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] result = new int[n];
        java.util.Stack<Integer> stack = new java.util.Stack<>(); // Store indices
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int index = stack.pop();
                result[index] = i - index;
            }
            stack.push(i);
        }
        
        return result;
    }
    
    // Largest Rectangle in Histogram
    // O(n) Time, O(n) Space
    private static int largestRectangleArea(int[] heights) {
        java.util.Stack<Integer> stack = new java.util.Stack<>();
        int maxArea = 0;
        int n = heights.length;
        
        for (int i = 0; i <= n; i++) {
            int currentHeight = (i == n) ? 0 : heights[i];
            
            while (!stack.isEmpty() && currentHeight < heights[stack.peek()]) {
                int height = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            
            stack.push(i);
        }
        
        return maxArea;
    }
    
    // Simplify Unix-style Path
    // O(n) Time, O(n) Space
    private static String simplifyPath(String path) {
        java.util.Stack<String> stack = new java.util.Stack<>();
        String[] components = path.split("/");
        
        for (String component : components) {
            if (component.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else if (!component.equals(".") && !component.isEmpty()) {
                stack.push(component);
            }
        }
        
        StringBuilder result = new StringBuilder();
        for (String dir : stack) {
            result.append("/").append(dir);
        }
        
        return result.length() > 0 ? result.toString() : "/";
    }

    // Queue Algorithm Implementations
    
    // Binary Tree Node for level order traversal
    static class BinaryTreeNode {
        int val;
        BinaryTreeNode left;
        BinaryTreeNode right;
        
        BinaryTreeNode(int val) {
            this.val = val;
        }
        
        BinaryTreeNode(int val, BinaryTreeNode left, BinaryTreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // Create a sample binary tree for testing
    private static BinaryTreeNode createSampleBinaryTree() {
        // Create tree:     3
        //                 / \
        //                9   20
        //                   /  \
        //                  15   7
        return new BinaryTreeNode(3,
            new BinaryTreeNode(9),
            new BinaryTreeNode(20,
                new BinaryTreeNode(15),
                new BinaryTreeNode(7)
            )
        );
    }
    
    // Binary Tree Level Order Traversal
    // O(n) Time, O(w) Space where w is maximum width
    private static List<List<Integer>> levelOrderTraversal(BinaryTreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        
        Queue<BinaryTreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                BinaryTreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            
            result.add(currentLevel);
        }
        
        return result;
    }
    
    // Sliding Window Maximum using Deque
    // O(n) Time, O(k) Space
    private static int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0) return new int[0];
        
        int n = nums.length;
        int[] result = new int[n - k + 1];
        java.util.Deque<Integer> deque = new java.util.ArrayDeque<>(); // Store indices
        
        for (int i = 0; i < n; i++) {
            // Remove indices outside the current window
            while (!deque.isEmpty() && deque.peekFirst() < i - k + 1) {
                deque.pollFirst();
            }
            
            // Remove indices whose corresponding values are smaller than current
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            
            deque.offerLast(i);
            
            // Add to result if window is complete
            if (i >= k - 1) {
                result[i - k + 1] = nums[deque.peekFirst()];
            }
        }
        
        return result;
    }
    
    // First Unique Character in String
    // O(n) Time, O(1) Space (26 letters)
    private static int firstUniqueChar(String s) {
        // Count frequency of each character
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        
        // Find first character with frequency 1
        for (int i = 0; i < s.length(); i++) {
            if (count[s.charAt(i) - 'a'] == 1) {
                return i;
            }
        }
        
        return -1;
    }
    
    // Hit Counter Design - Count hits in last 300 seconds
    static class HitCounter {
        private Queue<Integer> hits;
        
        public HitCounter() {
            hits = new LinkedList<>();
        }
        
        // Record a hit at timestamp
        // O(1) Time
        public void hit(int timestamp) {
            hits.offer(timestamp);
        }
        
        // Get hits count in past 300 seconds
        // O(n) Time where n is number of hits in past 300 seconds
        public int getHits(int timestamp) {
            // Remove hits older than 300 seconds
            while (!hits.isEmpty() && hits.peek() <= timestamp - 300) {
                hits.poll();
            }
            return hits.size();
        }
    }
    
    // Number of Islands using BFS
    // O(m*n) Time, O(min(m,n)) Space
    private static int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        
        int m = grid.length;
        int n = grid[0].length;
        int count = 0;
        
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    
                    // BFS to mark all connected land
                    Queue<int[]> queue = new LinkedList<>();
                    queue.offer(new int[]{i, j});
                    grid[i][j] = '0'; // Mark as visited
                    
                    while (!queue.isEmpty()) {
                        int[] cell = queue.poll();
                        int row = cell[0];
                        int col = cell[1];
                        
                        for (int[] dir : directions) {
                            int newRow = row + dir[0];
                            int newCol = col + dir[1];
                            
                            if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && grid[newRow][newCol] == '1') {
                                grid[newRow][newCol] = '0'; // Mark as visited
                                queue.offer(new int[]{newRow, newCol});
                            }
                        }
                    }
                }
            }
        }
        
        return count;
    }

    private static void bitManipulation() {
        // Bit Manipulation Algorithms
        // Fundamental operations for efficient computation and problem solving
        /*
         * Common Use Cases:
         * - Set/check/clear/toggle specific bits
         * - Check if number is power of 2
         * - Count number of 1 bits (population count)
         * - Find single number in array where others appear twice
         * - Subset generation using bit masks
         * - Fast multiplication/division by powers of 2
         */
        System.out.println("Bit Manipulation Algorithms:");
        
        // Basic Bit Operations
        int num = 12; // Binary: 1100
        System.out.println("Number: " + num + " (Binary: " + Integer.toBinaryString(num) + ")");
        
        // Check if bit at position i is set
        int position = 2;
        boolean isBitSet = (num & (1 << position)) != 0;
        System.out.println("Bit at position " + position + " is set: " + isBitSet);
        
        // Set bit at position i
        int setBit = num | (1 << position);
        System.out.println("After setting bit at position " + position + ": " + setBit + " (Binary: " + Integer.toBinaryString(setBit) + ")");
        
        // Clear bit at position i
        int clearBit = num & ~(1 << position);
        System.out.println("After clearing bit at position " + position + ": " + clearBit + " (Binary: " + Integer.toBinaryString(clearBit) + ")");
        
        // Toggle bit at position i
        int toggleBit = num ^ (1 << position);
        System.out.println("After toggling bit at position " + position + ": " + toggleBit + " (Binary: " + Integer.toBinaryString(toggleBit) + ")");
        
        // Check if number is power of 2
        // O(1) Time, O(1) Space
        int[] powerOfTwoTests = {1, 2, 3, 4, 8, 15, 16, 32};
        System.out.println("\nPower of 2 checks:");
        for (int n : powerOfTwoTests) {
            boolean isPowerOfTwo = isPowerOfTwo(n);
            System.out.println(n + " is power of 2: " + isPowerOfTwo);
        }
        
        // Count number of 1 bits (Hamming Weight)
        // O(number of 1 bits) Time, O(1) Space
        int[] hammingTests = {5, 11, 15, 128};
        System.out.println("\nCounting 1 bits:");
        for (int n : hammingTests) {
            int count = hammingWeight(n);
            System.out.println(n + " (Binary: " + Integer.toBinaryString(n) + ") has " + count + " one bits");
        }
        
        // Find single number (XOR trick)
        // O(n) Time, O(1) Space
        int[] singleNumberArray = {2, 2, 1, 3, 3, 4, 4};
        int singleNum = findSingleNumber(singleNumberArray);
        System.out.println("\nIn array " + java.util.Arrays.toString(singleNumberArray) + ", single number is: " + singleNum);
        
        // Generate all subsets using bit manipulation
        // O(2^n * n) Time, O(2^n * n) Space for output
        int[] subsetArray = {1, 2, 3};
        List<List<Integer>> allSubsets = generateSubsetsBitwise(subsetArray);
        System.out.println("\nAll subsets of " + java.util.Arrays.toString(subsetArray) + ":");
        for (List<Integer> subset : allSubsets) {
            System.out.println(subset);
        }
        
        // Reverse bits of a 32-bit integer
        // O(1) Time, O(1) Space
        int reverseTest = 43261596; // Binary: 00000010100101000001111010011100
        int reversed = reverseBits(reverseTest);
        System.out.println("\nReverse bits:");
        System.out.println("Original: " + reverseTest + " (Binary: " + String.format("%32s", Integer.toBinaryString(reverseTest)).replace(' ', '0') + ")");
        System.out.println("Reversed: " + reversed + " (Binary: " + String.format("%32s", Integer.toBinaryString(reversed)).replace(' ', '0') + ")");
        
        // Find missing number using XOR
        // O(n) Time, O(1) Space
        int[] missingArray = {3, 0, 1}; // Missing 2 from range [0, 3]
        int missingNum = findMissingNumber(missingArray);
        System.out.println("\nIn array " + java.util.Arrays.toString(missingArray) + ", missing number is: " + missingNum);
        
        // Fast multiplication and division by powers of 2
        int fastNum = 20;
        System.out.println("\nFast operations on " + fastNum + ":");
        System.out.println("Multiply by 4 (left shift 2): " + (fastNum << 2));
        System.out.println("Divide by 4 (right shift 2): " + (fastNum >> 2));
        System.out.println("Multiply by 8 (left shift 3): " + (fastNum << 3));
        System.out.println("Divide by 8 (right shift 3): " + (fastNum >> 3));
        
        // Get rightmost set bit
        int rightmostTest = 12; // Binary: 1100
        int rightmostSetBit = getRightmostSetBit(rightmostTest);
        System.out.println("\nRightmost set bit of " + rightmostTest + " (Binary: " + Integer.toBinaryString(rightmostTest) + ") is at position: " + rightmostSetBit);
        
        // Check if two numbers have opposite signs
        int num1 = 5, num2 = -3;
        boolean oppositeSigns = haveOppositeSigns(num1, num2);
        System.out.println("\nNumbers " + num1 + " and " + num2 + " have opposite signs: " + oppositeSigns);
        
        // Swap two numbers without using temporary variable
        int a = 10, b = 20;
        System.out.println("\nBefore swap: a = " + a + ", b = " + b);
        int[] swapped = swapWithoutTemp(a, b);
        System.out.println("After swap: a = " + swapped[0] + ", b = " + swapped[1]);
        
        System.out.println("\nBit Manipulation Complete");
    }
    
    // Check if number is power of 2
    // O(1) Time, O(1) Space
    private static boolean isPowerOfTwo(int n) {
        // A power of 2 has exactly one bit set
        // n & (n-1) removes the rightmost set bit
        // For power of 2, this should result in 0
        return n > 0 && (n & (n - 1)) == 0;
    }
    
    // Count number of 1 bits (Hamming Weight)
    // O(number of 1 bits) Time, O(1) Space
    private static int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            // Remove the rightmost set bit
            n = n & (n - 1);
            count++;
        }
        return count;
    }
    
    // Alternative implementation using Brian Kernighan's algorithm
    private static int hammingWeightAlternative(int n) {
        int count = 0;
        while (n != 0) {
            count += n & 1; // Add 1 if rightmost bit is set
            n >>>= 1; // Unsigned right shift
        }
        return count;
    }
    
    // Find single number where all others appear twice
    // O(n) Time, O(1) Space
    private static int findSingleNumber(int[] nums) {
        int result = 0;
        // XOR all numbers - duplicates will cancel out
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }
    
    // Generate all subsets using bit manipulation
    // O(2^n * n) Time, O(2^n * n) Space for output
    private static List<List<Integer>> generateSubsetsBitwise(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        int n = nums.length;
        int totalSubsets = 1 << n; // 2^n
        
        // Generate all possible bit masks from 0 to 2^n - 1
        for (int mask = 0; mask < totalSubsets; mask++) {
            List<Integer> subset = new ArrayList<>();
            
            // Check each bit position
            for (int i = 0; i < n; i++) {
                // If bit at position i is set, include nums[i]
                if ((mask & (1 << i)) != 0) {
                    subset.add(nums[i]);
                }
            }
            
            result.add(subset);
        }
        
        return result;
    }
    
    // Reverse bits of a 32-bit unsigned integer
    // O(1) Time, O(1) Space
    private static int reverseBits(int n) {
        int result = 0;
        
        for (int i = 0; i < 32; i++) {
            // Get the rightmost bit of n
            int bit = n & 1;
            
            // Shift result left and add the bit
            result = (result << 1) | bit;
            
            // Shift n right to process next bit
            n >>>= 1;
        }
        
        return result;
    }
    
    // Find missing number in array containing n distinct numbers in range [0, n]
    // O(n) Time, O(1) Space
    private static int findMissingNumber(int[] nums) {
        int n = nums.length;
        int expectedXor = 0;
        int actualXor = 0;
        
        // XOR all numbers from 0 to n
        for (int i = 0; i <= n; i++) {
            expectedXor ^= i;
        }
        
        // XOR all numbers in array
        for (int num : nums) {
            actualXor ^= num;
        }
        
        // Missing number is the XOR of expected and actual
        return expectedXor ^ actualXor;
    }
    
    // Get position of rightmost set bit (1-indexed)
    // O(1) Time, O(1) Space
    private static int getRightmostSetBit(int n) {
        if (n == 0) return -1; // No set bits
        
        // n & (-n) isolates the rightmost set bit
        int rightmostBit = n & (-n);
        
        // Count position (1-indexed)
        int position = 1;
        while (rightmostBit > 1) {
            rightmostBit >>= 1;
            position++;
        }
        
        return position;
    }
    
    // Check if two integers have opposite signs
    // O(1) Time, O(1) Space
    private static boolean haveOppositeSigns(int x, int y) {
        // XOR of two numbers with opposite signs will have MSB set
        return (x ^ y) < 0;
    }
    
    // Swap two numbers without using temporary variable
    // O(1) Time, O(1) Space
    private static int[] swapWithoutTemp(int a, int b) {
        // Using XOR swap
        a = a ^ b;
        b = a ^ b; // b = (a ^ b) ^ b = a
        a = a ^ b; // a = (a ^ b) ^ a = b
        
        return new int[]{a, b};
    }
    
    // Alternative swap using arithmetic (watch out for overflow)
    private static int[] swapArithmetic(int a, int b) {
        a = a + b;
        b = a - b; // b = (a + b) - b = a
        a = a - b; // a = (a + b) - a = b
        
        return new int[]{a, b};
    }
    
    // Check if a number is even or odd using bit manipulation
    // O(1) Time, O(1) Space
    private static boolean isEven(int n) {
        return (n & 1) == 0;
    }
    
    // Turn off the rightmost set bit
    // O(1) Time, O(1) Space
    private static int turnOffRightmostSetBit(int n) {
        return n & (n - 1);
    }
    
    // Turn on the rightmost unset bit
    // O(1) Time, O(1) Space
    private static int turnOnRightmostUnsetBit(int n) {
        return n | (n + 1);
    }
    
    // Check if only one bit is set (power of 2 check)
    // O(1) Time, O(1) Space
    private static boolean hasOnlyOneBitSet(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    private static void greedy() {
        // Greedy Algorithms
        // Make locally optimal choices at each step to achieve global optimum
        /*
         * Key Characteristics:
         * - Makes the best choice at each step without considering future consequences
         * - Works well when local optimal choices lead to global optimum
         * - Often used for optimization problems
         * - Generally faster than dynamic programming but doesn't always guarantee optimal solution
         * 
         * Common Use Cases:
         * - Activity/Interval scheduling
         * - Huffman coding, fractional knapsack
         * - Minimum spanning trees (Kruskal's, Prim's)
         * - Shortest path algorithms (Dijkstra's)
         * - Job scheduling, resource allocation
         */
        System.out.println("Greedy Algorithms:");
        
        // Example 1: Activity Selection Problem
        // O(n log n) Time due to sorting, O(1) Space
        // Select maximum number of non-overlapping activities
        int[][] activities = {{1, 3}, {2, 4}, {3, 5}, {0, 6}, {5, 7}, {8, 9}, {5, 9}};
        List<int[]> selectedActivities = activitySelection(activities);
        System.out.println("Activity Selection Problem:");
        System.out.println("Available activities (start, end): ");
        for (int[] activity : activities) {
            System.out.println("  " + java.util.Arrays.toString(activity));
        }
        System.out.println("Selected activities for maximum non-overlapping: ");
        for (int[] activity : selectedActivities) {
            System.out.println("  " + java.util.Arrays.toString(activity));
        }
        System.out.println("Total activities selected: " + selectedActivities.size());
        
        // Example 2: Fractional Knapsack Problem
        // O(n log n) Time, O(1) Space
        // Maximize value within weight constraint (can take fractions)
        Item[] items = {
            new Item(10, 60),  // value=60, weight=10, ratio=6.0
            new Item(20, 100), // value=100, weight=20, ratio=5.0
            new Item(30, 120)  // value=120, weight=30, ratio=4.0
        };
        int capacity = 50;
        double maxValue = fractionalKnapsack(items, capacity);
        System.out.println("\nFractional Knapsack Problem:");
        System.out.println("Items (weight, value):");
        for (Item item : items) {
            System.out.println("  Weight: " + item.weight + ", Value: " + item.value + ", Ratio: " + (item.value / (double)item.weight));
        }
        System.out.println("Knapsack capacity: " + capacity);
        System.out.println("Maximum value: " + maxValue);
        
        // Example 3: Coin Change (Greedy - works for certain coin systems)
        // O(n) Time where n is the amount, O(1) Space
        // Find minimum number of coins to make change
        int[] coins = {25, 10, 5, 1}; // US coin system - greedy works
        int amount = 67;
        List<Integer> coinChange = greedyCoinChange(coins, amount);
        System.out.println("\nGreedy Coin Change:");
        System.out.println("Coins available: " + java.util.Arrays.toString(coins));
        System.out.println("Amount to make: " + amount);
        System.out.println("Coins used: " + coinChange);
        System.out.println("Total coins: " + coinChange.size());
        
        // Example 4: Job Scheduling with Deadlines
        // O(n log n) Time, O(n) Space
        // Maximize profit by scheduling jobs within deadlines
        Job[] jobs = {
            new Job('A', 2, 100),
            new Job('B', 1, 19),
            new Job('C', 2, 27),
            new Job('D', 1, 25),
            new Job('E', 3, 15)
        };
        List<Job> scheduledJobs = jobScheduling(jobs);
        System.out.println("\nJob Scheduling with Deadlines:");
        System.out.println("Available jobs (ID, deadline, profit):");
        for (Job job : jobs) {
            System.out.println("  " + job.id + ": deadline=" + job.deadline + ", profit=" + job.profit);
        }
        System.out.println("Scheduled jobs for maximum profit:");
        int totalProfit = 0;
        for (Job job : scheduledJobs) {
            System.out.println("  " + job.id + ": profit=" + job.profit);
            totalProfit += job.profit;
        }
        System.out.println("Total profit: " + totalProfit);
        
        // Example 5: Minimum Number of Platforms
        // O(n log n) Time, O(1) Space
        // Find minimum platforms needed for train scheduling
        int[] arrivals = {900, 940, 950, 1100, 1500, 1800};
        int[] departures = {910, 1200, 1120, 1130, 1900, 2000};
        int minPlatforms = findMinimumPlatforms(arrivals, departures);
        System.out.println("\nMinimum Platforms Problem:");
        System.out.println("Train arrivals: " + java.util.Arrays.toString(arrivals));
        System.out.println("Train departures: " + java.util.Arrays.toString(departures));
        System.out.println("Minimum platforms needed: " + minPlatforms);
        
        // Example 6: Huffman Coding (Character Frequency Encoding)
        // O(n log n) Time, O(n) Space
        // Build optimal prefix codes based on character frequencies
        char[] characters = {'a', 'b', 'c', 'd', 'e', 'f'};
        int[] frequencies = {5, 9, 12, 13, 16, 45};
        HuffmanNode huffmanRoot = buildHuffmanTree(characters, frequencies);
        Map<Character, String> huffmanCodes = new HashMap<>();
        generateHuffmanCodes(huffmanRoot, "", huffmanCodes);
        System.out.println("\nHuffman Coding:");
        System.out.println("Character frequencies:");
        for (int i = 0; i < characters.length; i++) {
            System.out.println("  '" + characters[i] + "': " + frequencies[i]);
        }
        System.out.println("Huffman codes:");
        for (Map.Entry<Character, String> entry : huffmanCodes.entrySet()) {
            System.out.println("  '" + entry.getKey() + "': " + entry.getValue());
        }
        
        // Example 7: Gas Station Problem
        // O(n) Time, O(1) Space
        // Find starting gas station to complete circular tour
        int[] gas = {1, 2, 3, 4, 5};
        int[] cost = {3, 4, 5, 1, 2};
        int startStation = canCompleteCircuit(gas, cost);
        System.out.println("\nGas Station Problem:");
        System.out.println("Gas at stations: " + java.util.Arrays.toString(gas));
        System.out.println("Cost to next station: " + java.util.Arrays.toString(cost));
        if (startStation != -1) {
            System.out.println("Can complete circuit starting from station: " + startStation);
        } else {
            System.out.println("Cannot complete circuit");
        }
        
        // Example 8: Jump Game (Can reach end)
        // O(n) Time, O(1) Space
        // Check if you can reach the last index
        int[] jumpArray = {2, 3, 1, 1, 4};
        boolean canJump = canJumpToEnd(jumpArray);
        System.out.println("\nJump Game:");
        System.out.println("Array: " + java.util.Arrays.toString(jumpArray));
        System.out.println("Can reach end: " + canJump);
        
        // Jump Game II - Minimum jumps to reach end
        // O(n) Time, O(1) Space
        int minJumps = minimumJumps(jumpArray);
        System.out.println("Minimum jumps to reach end: " + minJumps);
        
        System.out.println("\nGreedy Algorithms Complete");
    }
    
    // Activity Selection Problem - Select maximum non-overlapping activities
    // O(n log n) Time due to sorting, O(1) Space
    private static List<int[]> activitySelection(int[][] activities) {
        List<int[]> result = new ArrayList<>();
        
        // Sort activities by end time
        java.util.Arrays.sort(activities, (a, b) -> a[1] - b[1]);
        
        // Select first activity
        result.add(activities[0]);
        int lastEndTime = activities[0][1];
        
        // Select subsequent non-overlapping activities
        for (int i = 1; i < activities.length; i++) {
            if (activities[i][0] >= lastEndTime) { // Start time >= last end time
                result.add(activities[i]);
                lastEndTime = activities[i][1];
            }
        }
        
        return result;
    }
    
    // Item class for Fractional Knapsack
    static class Item {
        int weight;
        int value;
        
        Item(int weight, int value) {
            this.weight = weight;
            this.value = value;
        }
    }
    
    // Fractional Knapsack Problem - Maximize value with weight constraint
    // O(n log n) Time, O(1) Space
    private static double fractionalKnapsack(Item[] items, int capacity) {
        // Sort items by value-to-weight ratio in descending order
        java.util.Arrays.sort(items, (a, b) -> Double.compare(
            (double)b.value / b.weight, (double)a.value / a.weight));
        
        double totalValue = 0.0;
        int remainingCapacity = capacity;
        
        for (Item item : items) {
            if (remainingCapacity >= item.weight) {
                // Take entire item
                totalValue += item.value;
                remainingCapacity -= item.weight;
            } else if (remainingCapacity > 0) {
                // Take fraction of item
                double fraction = (double)remainingCapacity / item.weight;
                totalValue += item.value * fraction;
                remainingCapacity = 0;
                break;
            }
        }
        
        return totalValue;
    }
    
    // Greedy Coin Change (works for certain coin systems like US coins)
    // O(n) Time where n is the amount, O(1) Space
    private static List<Integer> greedyCoinChange(int[] coins, int amount) {
        List<Integer> result = new ArrayList<>();
        
        for (int coin : coins) {
            while (amount >= coin) {
                result.add(coin);
                amount -= coin;
            }
        }
        
        return result;
    }
    
    // Job class for Job Scheduling
    static class Job {
        char id;
        int deadline;
        int profit;
        
        Job(char id, int deadline, int profit) {
            this.id = id;
            this.deadline = deadline;
            this.profit = profit;
        }
    }
    
    // Job Scheduling with Deadlines - Maximize profit
    // O(n log n) Time, O(n) Space
    private static List<Job> jobScheduling(Job[] jobs) {
        // Sort jobs by profit in descending order
        java.util.Arrays.sort(jobs, (a, b) -> b.profit - a.profit);
        
        // Find maximum deadline to determine schedule array size
        int maxDeadline = 0;
        for (Job job : jobs) {
            maxDeadline = Math.max(maxDeadline, job.deadline);
        }
        
        // Schedule array to track which jobs are scheduled at each time slot
        Job[] schedule = new Job[maxDeadline];
        boolean[] occupied = new boolean[maxDeadline];
        
        List<Job> result = new ArrayList<>();
        
        // Try to schedule each job
        for (Job job : jobs) {
            // Find a free slot for this job (before its deadline)
            for (int slot = Math.min(maxDeadline - 1, job.deadline - 1); slot >= 0; slot--) {
                if (!occupied[slot]) {
                    occupied[slot] = true;
                    schedule[slot] = job;
                    result.add(job);
                    break;
                }
            }
        }
        
        return result;
    }
    
    // Minimum Platforms needed for train scheduling
    // O(n log n) Time, O(1) Space
    private static int findMinimumPlatforms(int[] arrivals, int[] departures) {
        java.util.Arrays.sort(arrivals);
        java.util.Arrays.sort(departures);
        
        int platforms = 0;
        int maxPlatforms = 0;
        int i = 0, j = 0;
        
        // Use two pointers to simulate events
        while (i < arrivals.length && j < departures.length) {
            if (arrivals[i] <= departures[j]) {
                // Train arrives, need one more platform
                platforms++;
                maxPlatforms = Math.max(maxPlatforms, platforms);
                i++;
            } else {
                // Train departs, free one platform
                platforms--;
                j++;
            }
        }
        
        return maxPlatforms;
    }
    
    // Huffman Tree Node for encoding
    static class HuffmanNode {
        char character;
        int frequency;
        HuffmanNode left;
        HuffmanNode right;
        
        HuffmanNode(char character, int frequency) {
            this.character = character;
            this.frequency = frequency;
        }
        
        HuffmanNode(int frequency, HuffmanNode left, HuffmanNode right) {
            this.character = '\0'; // Internal node
            this.frequency = frequency;
            this.left = left;
            this.right = right;
        }
    }
    
    // Build Huffman Tree for optimal character encoding
    // O(n log n) Time, O(n) Space
    private static HuffmanNode buildHuffmanTree(char[] characters, int[] frequencies) {
        // Priority queue (min heap) based on frequency
        Queue<HuffmanNode> pq = new java.util.PriorityQueue<>((a, b) -> a.frequency - b.frequency);
        
        // Add all characters to priority queue
        for (int i = 0; i < characters.length; i++) {
            pq.offer(new HuffmanNode(characters[i], frequencies[i]));
        }
        
        // Build tree by combining nodes with lowest frequencies
        while (pq.size() > 1) {
            HuffmanNode left = pq.poll();
            HuffmanNode right = pq.poll();
            
            HuffmanNode merged = new HuffmanNode(
                left.frequency + right.frequency, left, right);
            
            pq.offer(merged);
        }
        
        return pq.poll(); // Root of Huffman tree
    }
    
    // Generate Huffman codes from tree
    private static void generateHuffmanCodes(HuffmanNode root, String code, Map<Character, String> codes) {
        if (root == null) return;
        
        // Leaf node - store the code
        if (root.left == null && root.right == null) {
            codes.put(root.character, code.isEmpty() ? "0" : code);
            return;
        }
        
        // Traverse left and right
        generateHuffmanCodes(root.left, code + "0", codes);
        generateHuffmanCodes(root.right, code + "1", codes);
    }
    
    // Gas Station Problem - Find starting station to complete circuit
    // O(n) Time, O(1) Space
    private static int canCompleteCircuit(int[] gas, int[] cost) {
        int totalGas = 0;
        int totalCost = 0;
        int currentGas = 0;
        int start = 0;
        
        for (int i = 0; i < gas.length; i++) {
            totalGas += gas[i];
            totalCost += cost[i];
            currentGas += gas[i] - cost[i];
            
            // If we can't reach the next station, start from next station
            if (currentGas < 0) {
                start = i + 1;
                currentGas = 0;
            }
        }
        
        // Check if total gas is enough to complete the circuit
        return totalGas >= totalCost ? start : -1;
    }
    
    // Jump Game - Check if can reach the end
    // O(n) Time, O(1) Space
    private static boolean canJumpToEnd(int[] nums) {
        int maxReach = 0;
        
        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) return false; // Can't reach this position
            
            maxReach = Math.max(maxReach, i + nums[i]);
            
            if (maxReach >= nums.length - 1) return true; // Can reach end
        }
        
        return false;
    }
    
    // Jump Game II - Minimum jumps to reach end
    // O(n) Time, O(1) Space
    private static int minimumJumps(int[] nums) {
        if (nums.length <= 1) return 0;
        
        int jumps = 0;
        int currentEnd = 0;
        int farthest = 0;
        
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            
            // If we've reached the end of current jump
            if (i == currentEnd) {
                jumps++;
                currentEnd = farthest;
                
                // If we can reach the end from current position
                if (currentEnd >= nums.length - 1) break;
            }
        }
        
        return jumps;
    }
}