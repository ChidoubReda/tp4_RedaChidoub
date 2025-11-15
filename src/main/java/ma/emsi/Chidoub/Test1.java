package ma.emsi.Chidoub;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;

public class Test1 {

    public static void main(String[] args) {

        System.out.println("\n--- TEST 1 : RAG test1 (manuel, Gemini + text-embedding-004) ---\n");

        // 1) Charger le PDF avec Apache Tika
        DocumentParser parser = new ApacheTikaDocumentParser();
        Path pdf = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(pdf, parser);

        // 2) Découper en fragments (chunks)
        // Découpage plus fin → meilleur rappel
        List<TextSegment> fragments = DocumentSplitters
                .recursive(150, 30)
                .split(document);

        // 3) Modèle d'embeddings (Google text-embedding-004, comme dans Test4)
        String key = System.getenv("GEMINI_API_KEY");
        if (key == null) {
            throw new IllegalStateException("Définis la variable d'environnement GEMINI_API_KEY.");
        }

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(key)
                .modelName("text-embedding-004")
                .build();

        // 4) Générer les embeddings pour tous les fragments
        List<Embedding> vectors = embeddingModel.embedAll(fragments).content();

        // 5) Stockage en RAM (base vectorielle en mémoire)
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, fragments);

        // 6) Modèle de chat Gemini (LLM)
        // Température basse (≤ 0.3) pour plus de précision, comme dans ton Test4
        ChatModel llm = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.0-flash")
                .temperature(0.2)
                .timeout(Duration.ofSeconds(60))
                .build();

        // 7) Récupérateur de contexte (RAG)
        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.1)
                .build();

        // 8) Mémoire conversationnelle (on garde les 10 derniers messages)
        var memory = MessageWindowChatMemory.withMaxMessages(10);

        // 9) Assistant RAG (implémentation générée par LangChain4j)
        Assistant bot = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .contentRetriever(retriever)
                .chatMemory(memory)
                .build();

        // 10) Console interactive
        try (Scanner sc = new Scanner(System.in)) {
            System.out.println("Pose une question sur le PDF (tape 'exit' pour quitter) :");
            while (true) {
                System.out.print("> ");
                String q = sc.nextLine().trim();

                if (q.equalsIgnoreCase("exit")) break;
                if (q.isEmpty()) continue; // ignore input vide

                String answer = bot.chat(q);
                System.out.println("\n" + answer + "\n");
            }
        }
    }
}
