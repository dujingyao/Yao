#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义飞机票预订信息结构体
typedef struct Ticket {
    int id;                // 飞机票ID
    char name[50];         // 乘客姓名
    char flight_number[10]; // 航班号
    struct Ticket* next;   // 指向下一个节点的指针
} Ticket;

// 初始化链表
Ticket* initList() {
    return NULL;
}

// 创建新节点
Ticket* createTicket(int id, const char* name, const char* flight_number) {
    Ticket* new_ticket = (Ticket*)malloc(sizeof(Ticket));
    if (new_ticket != NULL) {
        new_ticket->id = id;
        strncpy(new_ticket->name, name, sizeof(new_ticket->name) - 1);
        strncpy(new_ticket->flight_number, flight_number, sizeof(new_ticket->flight_number) - 1);
        new_ticket->next = NULL;
    }
    return new_ticket;
}

// 插入节点到链表头部
void insertAtHead(Ticket** head, Ticket* new_ticket) {
    if (new_ticket != NULL) {
        new_ticket->next = *head;
        *head = new_ticket;
    }
}

// 删除节点
void deleteTicket(Ticket** head, int id) {
    Ticket* current = *head;
    Ticket* prev = NULL;
    while (current != NULL && current->id != id) {
        prev = current;
        current = current->next;
    }
    if (current == NULL) {
        printf("Ticket with ID %d not found.\n", id);
        return;
    }
    if (prev == NULL) {
        *head = current->next;
    } else {
        prev->next = current->next;
    }
    free(current);
}

// 查找节点
Ticket* findTicket(Ticket* head, int id) {
    while (head != NULL) {
        if (head->id == id) {
            return head;
        }
        head = head->next;
    }
    return NULL;
}

// 取值
void getValue(Ticket* ticket) {
    if (ticket != NULL) {
        printf("ID: %d, Name: %s, Flight Number: %s\n", ticket->id, ticket->name, ticket->flight_number);
    } else {
        printf("Ticket not found.\n");
    }
}

// 遍历链表
void traverseList(Ticket* head) {
    Ticket* current = head;
    while (current != NULL) {
        getValue(current);
        current = current->next;
    }
}

int main() {
    Ticket* head = initList();

    // 建立链表
    insertAtHead(&head, createTicket(1, "Alice", "FR123"));
    insertAtHead(&head, createTicket(2, "Bob", "FR456"));
    insertAtHead(&head, createTicket(3, "Charlie", "FR789"));

    // 遍历链表
    printf("Initial list:\n");
    traverseList(head);

    // 插入新节点
    insertAtHead(&head, createTicket(4, "Dave", "FR101"));
    printf("\nList after insertion:\n");
    traverseList(head);

    // 删除节点
    deleteTicket(&head, 2);
    printf("\nList after deletion:\n");
    traverseList(head);

    // 查找节点
    Ticket* found_ticket = findTicket(head, 3);
    printf("\nFound ticket:\n");
    getValue(found_ticket);

    // 取值
    printf("\nValue of the first ticket:\n");
    getValue(head);

    // 清理链表
    while (head != NULL) {
        Ticket* temp = head;
        head = head->next;
        free(temp);
    }

    return 0;
}