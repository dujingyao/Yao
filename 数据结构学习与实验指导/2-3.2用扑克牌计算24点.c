#include<stdio.h>
int cal(int a,int b,char c){
    switch(c){
        case '+':return a+b;
        case '-':return a-b;
        case '*':return a*b;
        case '/':return (b!=0&&a%b==0)?a/b:0;
        default:return 0;
    }
}
int main(){
    int a[4];
    scanf("%d %d %d %d",&a[0],&a[1],&a[2],&a[3]);
    int c[24][4];
    char b[4]={'+','-','*','/'};
    int a0,a1,a2,a3,i=0;
    for(a0=0;a0<4;a0++){
        for(a1=0;a1<4;a1++){
            for(a2=0;a2<4;a2++){
                for(a3=0;a3<4;a3++){
                    if(a0==a1||a0==a2||a0==a3||a1==a2||a1==a3||a2==a3) continue;
                    c[i][0]=a[a0];
                    c[i][1]=a[a1];
                    c[i][2]=a[a2];
                    c[i][3]=a[a3];
                    i++;    
                }
            }
        }
    }
    char d[64][3];
    int d0,d1,d2;
    int j=0;
    for(j=0;j<64;j++){
        for(d0=0;d0<4;d0++){
            for(d1=0;d1<4;d1++){
                for(d2=0;d2<4;d2++){
                        d[j][0]=b[d0];
                        d[j][1]=b[d1];
                        d[j][2]=b[d2];
                        j++;
                }
            }
        }
    }
    int f=0;
    for(i=0;i<24;i++){
        for(j=0;j<64;j++){
            if(cal(cal(cal(c[i][0],c[i][1],d[j][0]),c[i][2],d[j][1]),c[i][3],d[j][2])==24){
                printf("((%d%c%d)%c%d)%c%d",c[i][0],d[j][0],c[i][1],d[j][1],c[i][2],d[j][2],c[i][3]);
                f=1;
                break;
            }      //第一种括号匹配
            if(cal(cal(c[i][0],c[i][1],d[j][0]),cal(c[i][2],c[i][3],d[j][2]),d[j][1])==24){
                printf("(%d%c%d)%c(%d%c%d)",c[i][0],d[j][0],c[i][1],d[j][1],c[i][2],d[j][2],c[i][3]);
                f=1;
                break;
            }     //第二种括号匹配
            if(cal(cal(c[i][0],cal(c[i][1],c[i][2],d[j][1]),d[j][0]),c[i][3],d[j][2])==24){
                printf("(%d%c(%d%c%d))%c%d",c[i][0],d[j][0],c[i][1],d[j][1],c[i][2],d[j][2],c[i][3]);
                f=1;
                break;
            }     //第三种括号匹配
            if(cal(c[i][0],cal(cal(c[i][1],c[i][2],d[j][1]),c[i][3],d[j][2]),d[j][0])==24){
                printf("%d%c((%d%c%d)%c%d)",c[i][0],d[j][0],c[i][1],d[j][1],c[i][2],d[j][2],c[i][3]);
                f=1;
                break;
            }    //第四种括号匹配
            if(cal(c[i][0],cal(c[i][1],cal(c[i][2],c[i][3],d[j][2]),d[j][1]),d[j][0])==24){
                printf("%d%c(%d%c(%d%c%d))",c[i][0],d[j][0],c[i][1],d[j][1],c[i][2],d[j][2],c[i][3]);
                f=1;
                break;
            }    //第五种括号匹配
        }
        if(f==1) break;
    }
    if(f==0){
        printf("-1\n");
    }
    return 0;
}