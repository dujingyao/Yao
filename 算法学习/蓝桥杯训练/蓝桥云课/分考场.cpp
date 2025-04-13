#include<bits/stdc++.h>
using namespace std;
const int N=110;
bool g[N][N];//存图
int n,m;
int p[N][N];//考场状态,p[j][k]=y:表示第j个考场的第k个座位，坐第y个人
int num=N;
void dfs(int x,int room){
	if(room>=num) return;//不符合
	if(x>n){//已经安排了n个人，结束 
		num=min(num,room);//更新最优解
		return; 
	}
	int j,k;
	for(j=1;j<=room;j++){//遍历教室 
		k=1;
		//遍历教室里的位置 
		while(p[j][k]&&!g[x][p[j][k]]){//如果j考场第k个位置有人坐并且二者不认识 
			k++;//去坐下一个位置 
		}
		if(p[j][k]==0){//如果没人坐 
			p[j][k]=x;
			dfs(x+1,room);
			p[j][k]=0;
		}
	}
	//如果安排不了，那就加一个考场
	p[room+1][1]=x;
	dfs(x+1,room+1);
	p[room+1][1]=0; 
}
int main(){
	cin>>n>>m;
	int a,b;
	for(int i=1;i<=m;i++){
		cin>>a>>b;
		g[a][b]=true;
		g[b][a]=true;
	}		
	dfs(1,1);
	cout<<num<<endl;
	return 0;
} 