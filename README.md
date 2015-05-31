waifu2x-converter-glsl
========================

waifu2x-converter-cpp (https://github.com/WL-Amigo/waifu2x-converter-cpp) をベースに  
OpenGLを使いGPUで計算処理を行うように改造したものです。  
ノートPCの非力なIntel製GPUでもたぶん動きます。  

exeバイナリのダウンロードはこちら↓  
https://github.com/ueshita/waifu2x-converter-glsl/releases

コマンドライン周りはwaifu2x-converter.exeのままですので、  
waifu2x-converter-glsl.exeをwaifu2x-converter.exeにリネームすれば  
GUIフロントエンドがそのまま使えると思います。  

動作環境：OpenGL 3.1が動作すること。  
（Intel HD Graphics 5000で動作確認済）  
※バイナリは32bitなので、そんなに大きな画像は処理できません。

References
========================

- Original implementation: https://github.com/nagadomi/waifu2x
- Python implementation: https://marcan.st/transf/waifu2x.py
- C++ implementation: https://github.com/WL-Amigo/waifu2x-converter-cpp
