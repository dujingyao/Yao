#include<stdio.h>
#include<string.h>

//将数组c中前n小的数的下标存入d
void f(int c[],int d[],int n,int start,int end){
    int i,j=0,x=0;
    while(1){
        for(i=start;i<end;i++){
            if(c[i]==x){
                d[j]=i;
                j++;
                n--;
                if(n==0) break;
            }
        }
        if(n==0) break;
        x++;
    }
    d[j]=-1;
}

int main(){
    int D,P;//D是海盗人数,P是宝石数
    scanf("%d %d",&P,&D);
    int a[D-1][D],b[D];
    for(int i=0;i<D-1;i++){
        for(int j=0;j<D;j++){
            a[i][j]=0;
            b[j]=0;
        }
    }
    int i,j,sum;
    a[0][D-2]=0;
    a[0][D-1]=P;
    for(i=1;i<D-1;i++){
        for(j=D-1-i;j<D;j++){
            sum=0;
            int n;
            if((D-j)%2==0) n=(D-j)/2;
            else n=(D-j)/2+1;
            f(a[i-1],b,n,j,D);
            for(int z=0;b[z]!=-1;z++){
                a[i][b[z]]=a[i-1][b[z]]+1;
                sum+=a[i][b[z]];
            }
            a[i][j-1]=P-sum;
            break;
        }
    }
    for(i=0;i<D;i++){
        printf("%d ",a[D-2][i]);
    }
    return 0;
}