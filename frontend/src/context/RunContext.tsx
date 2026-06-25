import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

interface RunContextValue {
  currentRunId: string | null;
  setCurrentRunId: (id: string | null) => void;
  queryList: string[];
  setQueryList: (queries: string[]) => void;
}

const RunContext = createContext<RunContextValue | null>(null);

export function RunProvider({ children }: { children: ReactNode }) {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [queryList, setQueryList] = useState<string[]>([]);

  return (
    <RunContext.Provider value={{ currentRunId, setCurrentRunId, queryList, setQueryList }}>
      {children}
    </RunContext.Provider>
  );
}

export function useRun(): RunContextValue {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used inside RunProvider");
  return ctx;
}
