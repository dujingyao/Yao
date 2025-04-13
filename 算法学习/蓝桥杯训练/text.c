#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSTATES 50000  // 定义最大状态数，即可能遇到的不同状态的个数
#define QUEUESIZE 100000 // 定义队列的最大容量，即队列中最多可以存储的状态数

// 状态结构体，包含当前状态和从初始状态到当前状态的步数
typedef struct {
    char state[10]; // 用一个长度为10的字符串来表示状态，包括9个字符和一个终止符'\0'
    int dist;       // 记录从初始状态到当前状态的步数
} State;

State queue[QUEUESIZE]; // 队列数组，用于存储待处理的状态
int front = 0, rear = 0; // 队列的前端和后端索引，用于实现队列操作

// 检查当前状态是否为目标状态
int isTarget(const char* state, const char* target) {
    return strcmp(state, target) == 0; // 如果当前状态与目标状态相同，返回1，否则返回0
}

// 检查移动是否有效，并返回空格的索引
int isValidMove(const char* state, int spaceIndex, int dir) {
    // 计算移动后的新索引
    int newRow = spaceIndex / 3 + (dir == 1 ? -1 : (dir == 2 ? 1 : 0));
    int newCol = spaceIndex % 3 + (dir == 3 ? -1 : (dir == 4 ? 1 : 0));
    // 检查新索引是否在范围内
    if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
        int newIndex = newRow * 3 + newCol;
        return newIndex; // 返回新的索引
    }
    return -1; // 如果移动无效，返回-1
}
 
// 添加新状态到队列
void enqueue(const char* state, int dist) {
    if (rear >= QUEUESIZE) return; // 如果队列满了，直接返回
    strcpy(queue[rear].state, state); // 复制状态到队列中
    queue[rear].dist = dist;         // 设置步数
    rear++;                          // 移动队列的后端索引
}

// 从队列中取出状态
State dequeue() {
    State result = queue[front]; // 获取队列前端的元素
    front++;                    // 移动队列的前端索引
    return result;              // 返回取出的状态
}

// 主函数，执行广度优先搜索（BFS）算法
int bfs(const char* start, const char* target) {
    char visited[MAXSTATES][10] = {0}; // 用于记录已访问的状态
    int spaceIndex = strchr(start, '.') - start; // 找到空格的索引

    enqueue(start, 0); // 将初始状态加入队列
    strcpy(visited[0], start); // 标记初始状态为已访问

    int directions[4] = {1, 2, 3, 4}; // 移动方向：上、下、左、右
    int v = 1; // 已访问状态的计数器

    while (front < rear) { // 当队列不为空时
        State current = dequeue(); // 取出队列中的第一个状态
        if (isTarget(current.state, target)) return current.dist; // 如果找到目标状态，返回步数

        // 尝试所有可能的移动
        for (int i = 0; i < 4; i++) {
            int newIndex = isValidMove(current.state, spaceIndex, directions[i]);
            if (newIndex != -1) { // 如果移动有效
                char newState[10]; // 创建新的状态
                strcpy(newState, current.state); // 复制当前状态
                // 交换空格和新位置的字符
                newState[spaceIndex] = newState[newIndex];
                newState[newIndex] = '.';

                // 检查新状态是否已访问
                int alreadyVisited = 0;
                for (int j = 0; j < v; j++) {
                    if (strcmp(visited[j], newState) == 0) {
                        alreadyVisited = 1;
                        break;
                    }
                }

                if (!alreadyVisited) { // 如果新状态未被访问
                    strcpy(visited[v], newState); // 标记为已访问
                    v++; // 增加已访问状态计数器
                    enqueue(newState, current.dist + 1); // 将新状态加入队列
                }
            }
        }
    }

    return -1; 
    }