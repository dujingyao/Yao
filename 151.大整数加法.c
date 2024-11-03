#include<stdio.h>
#include<string.h>
int main(){
    int T;
    scanf("%d",&T);
    char str1[1000],str2[1000];
    int strs1[1000],strs2[1000],strs[1000],n,b1,b2,k,t;
    while(T--){
        n=0;
         memset(strs1,0,sizeof(strs1));
         memset(strs2,0,sizeof(strs2));
         memset(strs,0,sizeof(strs));
         scanf("%s%s",str1,str2);
        b1=strlen(str1);
        b2=strlen(str2);
        if(b1>b2){
            n=b1-b2;
            for(int i=0;i<b1;i++){
                strs1[i]=str1[i]-'0';
                if(i<b2) strs2[i+n]=str2[i]-'0';
            }
        }
        else{
            n=b2-b1;
            for(int i=0;i<b2;i++){
                strs2[i]=str2[i]-'0';
                if(i<b2) strs1[i+n]=str1[i]-'0';
            }
        }
        if(b2>b1){
            t=b1;
            b1=b2;
            b2=t;
        }
        n=0;
        for(int i=b1-1;i>=0;i--){
                k=strs1[i]+strs2[i]+n;
                if(k>=10){
                    n=1;
                    strs[i]=k-10;
                }else{
                    n=0;
                    strs[i]=k;
                }
                if(i==0&&n==1) printf("1");
        }
        for(int i=0;i<b1;i++){
            printf("%d",strs[i]);
        }
        printf("\n");
    }

    return 0;
}