package ma.emsi.Chidoub.test1;

import dev.langchain4j.service.SystemMessage;

public interface Assistant {

    @SystemMessage(
            "Tu es un assistant RAG. Tu dois répondre uniquement à partir du PDF fourni. " +
                    "Si l'information n'est pas dans le document, dis que tu ne sais pas."
    )
    String chat(String userMessage);
}