#include<stdio.h>

int main(){
    int Sum,X;
    char OP;
    scanf("%d%c",&Sum,&OP);
    while(OP!='='){
        scanf("%d",&X);
        switch(OP){
            case '+':
                Sum+=X;
                break;
            case '-':
                Sum-=X;
                break;
            case '*':
                Sum*=X;
                break;
            case '/':
                if(X==0){
                    printf("ERROR\n");
                    return 0;
                }
                Sum/=X;
                break;
            default:
                printf("ERROR\n");
                return 0;
        }
        scanf("%c",&OP);
    }
    printf("%d\n",Sum);
    return 0;
}