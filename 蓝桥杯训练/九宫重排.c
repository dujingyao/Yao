#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define MAX_STATES 50000
#define QUEUE_SIZE 100000

typedef struct{
    char state[10];
    int dist;
}State;
State queue[QUEUE_SIZE];
int front=0,rear=0;

void check(const char*state,const char* target){
    return strcmp(state,target)==0;
}


int main(){
    

    return 0;
}