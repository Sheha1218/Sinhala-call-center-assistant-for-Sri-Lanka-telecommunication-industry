import asyncio
import edge_tts
import pyaudio

text ="Sim change එකක් කරගනිද්දි original customer brunch එකට visit කරන්න ඔනේ. අනිවර්යෙන්ම original customer NIC හො Driving license එක අරගෙන brunch එකට යන්න ඕනේ. අපි හිතමු NIC එක කියලා unclear හරි dammage අපිට NIC එක verified කරන්න බෑ .ඒ විතරක් නෙමෙයි NIC එක නැත්නම් අපි Driving license  එක ගෙනියන්ව කියමුකො එකගෙනිච්චත්  same එත් එක clear තියෙන්න ඔනේ. එක expired වෙලා තියෙන්න බෑ incase expire වෙලානම් අපි එක accept  කරන්නෙත් නැ. ඔය දෙකම ලග නැත්නම් අපිට එයාට option එකක් දෙන්න පුලුවන් ඒ තමය් Passport එක. ඒත් එකත් expired වෙලා තියෙන්න බෑ. තව perority customer කෙනෙක් නෙමෙයි නම් රුපියල් 100ක් change වෙනවා ඒ වගේම තමන්ගි නමට SIM එක තියෙන්නත් ඔනේ සර් . Sim එක active වෙන්න පැය 3ක් වගේ යනවා."
    
async def speak():
    communicate = edge_tts.Communicate(text=text,voice='si-LK-SameeraNeural')

    
    p =  pyaudio.PyAudio()
    stream =None
    
    async for chunk in communicate.stream():
        if chunk['type']=='audio':
            if stream is None:
                steam =p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=24000,
                              output=True)
                
                stream.write(chunk['data'])
                
    if steam:
        steam.stop_stream()
        stream.close()
    p.terminate()
    
asyncio.run(speak())