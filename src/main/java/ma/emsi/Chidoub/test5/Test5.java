package ma.emsi.Chidoub.test5;

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

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;

import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5 {

    public static void main(String[] args) {

        System.out.println("\n--- TEST 5 : RAG PDF + Web (Tavily) ---\n");

        configureLogger();

        // 1) Clés API
        String geminiKey = System.getenv("GEMINI_API_KEY");
        if (geminiKey == null) {
            throw new IllegalStateException("⚠️ Variable d'environnement GEMINI_API_KEY manquante");
        }

        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null) {
            throw new IllegalStateException("⚠️ Variable d'environnement TAVILY_API_KEY manquante");
        }

        // 2) Charger le PDF (RAG naïf de base)
        DocumentParser parser = new ApacheTikaDocumentParser();
        // adapte le chemin si besoin : rag.pdf / support_rag.pdf…
        Document document = FileSystemDocumentLoader.loadDocument(
                Paths.get("src/main/resources/rag.pdf"), parser);

        // 3) Découpage en segments
        List<TextSegment> segments = DocumentSplitters
                .recursive(200, 40)
                .split(document);

        // 4) Embeddings (Google text-embedding-004)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(geminiKey)
                .modelName("text-embedding-004")
                .build();

        List<Embedding> vectors = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, segments);

        // 5) ContentRetriever sur le PDF (comme dans le RAG naïf)
        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.2)
                .build();

        // 6) WebSearchEngine Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // 7) ContentRetriever pour le Web (Tavily)
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // 8) Modèle Gemini (LLM)
        ChatModel llm = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .timeout(Duration.ofSeconds(60))
                .logRequestsAndResponses(true)
                .build();

        // 9) QueryRouter : utiliser les 2 retrievers (PDF + Web)
        QueryRouter router = new DefaultQueryRouter(pdfRetriever, webRetriever);

        // 10) RetrievalAugmentor basé sur ce QueryRouter
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // 11) Assistant RAG + Web
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // 12) Boucle interactive
        try (Scanner sc = new Scanner(System.in)) {
            System.out.println("Pose une question (le système utilisera le PDF + le Web).");
            System.out.println("Tape 'exit' pour quitter.\n");

            while (true) {
                System.out.print("> ");
                String q = sc.nextLine().trim();
                if (q.equalsIgnoreCase("exit")) break;
                if (q.isEmpty()) continue;

                String answer = assistant.answer(q);
                System.out.println("\n🤖 " + answer + "\n");
            }
        }
    }

    private static void configureLogger() {
        Logger log = Logger.getLogger("dev.langchain4j");
        log.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        if (log.getHandlers().length == 0) {
            log.addHandler(handler);
        }
    }
}
